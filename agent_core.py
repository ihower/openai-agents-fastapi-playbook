from dataclasses import dataclass
from pydantic import BaseModel, Field
from agents import Agent, Runner, function_tool, RunContextWrapper, ModelSettings, set_default_openai_client, WebSearchTool, FileSearchTool
from tavily import AsyncTavilyClient
from datetime import datetime
from openai import AsyncOpenAI
import braintrust
import os
import asyncio
import aiosqlite
import json
import pathlib

ROOT_DIR = pathlib.Path(__file__).parent.absolute()

braintrust_logger = None
openai_client = None

# Load prompts from files
_prompt_cache = {}
def load_prompt(key: str) -> str:
    if key not in _prompt_cache:
        print(f"Loading prompt: {key}")
        prompt_path = ROOT_DIR / "prompts" / f"{key}.md"
        with open(prompt_path, "r", encoding="utf-8") as f:
            _prompt_cache[key] = f.read()
    return _prompt_cache[key]

async def get_previous_items(db, thread_id: str) -> tuple[list, dict]:
    """
    根據 thread_id 從 agent_turns 取得最後一筆對話的 raw_items 和 metadata
    如果沒有找到，返回空列表和空字典
    """
    input_items = []
    metadata = {}

    async with db.execute(
        "SELECT raw_items, metadata FROM agent_turns WHERE thread_id = ? ORDER BY id DESC LIMIT 1",
        (thread_id,)
    ) as cursor:
        row = await cursor.fetchone()

        if row and row[0]:
            try:
                raw_items_list = json.loads(row[0])
                for item_str in raw_items_list:
                    parsed_item = json.loads(item_str)
                    input_items.append(parsed_item)
                print(f"Loaded {len(input_items)} items from previous conversation")

                if row[1]:
                    metadata = json.loads(row[1])
            except Exception as e:
                print(f"Error loading raw_items: {e}")

    return input_items, metadata

# Context Engineering 閾值設定
TOOL_CALL_OUTPUT_TRIM_THRESHOLD = 150000  # 當 tokens 超過此值時，簡化 function_call_output
COMPACTION_THRESHOLD = 200000  # 當 tokens 超過此值時，呼叫 Compaction API 壓縮對話
COMPACTION_MODEL = "gpt-5.6-luna"  # 執行壓縮的模型，建議與 lead agent 相同

async def context_editing(input_items: list, used_tokens: int) -> list:
    """
    對 input_items 進行 context engineering，根據 token 使用情況進行剪裁

    Args:
        input_items: 要處理的對話記錄
        used_tokens: 前一次對話使用的 tokens 數量

    Returns:
        處理後的 input_items
    """
    print(f"previous used_tokens: {used_tokens}")

    # Context Engineering 1: Tool call output trimming
    # 當 tokens 超過閾值時，簡化 function_call_output 內容
    if used_tokens > TOOL_CALL_OUTPUT_TRIM_THRESHOLD:
        print(f"Trigger tool call output filter: used_tokens={used_tokens}")
        for item in input_items:
            if item.get("type") == "function_call_output":
                print(" remove function_call_output! ")
                item["output"] = "Tool results removed (context limit). Re-run the tool if needed."

    # Context Engineering 2: Compaction
    # 當 tokens 超過閾值時，呼叫 Compaction API 讓模型把整段對話壓縮成摘要
    # 相較於直接移除最舊的 turns，壓縮後仍能保留早期對話的重點資訊
    if used_tokens > COMPACTION_THRESHOLD:
        print(f"Trigger message compaction: used_tokens={used_tokens}")

        compacted = await openai_client.responses.compact(
            model=COMPACTION_MODEL,
            input=input_items,
        )
        input_items = [item.model_dump(exclude_none=True) for item in compacted.output]

        print(f"Compaction done: {len(input_items)} items after compaction")

    return input_items

async def save_agent_turn(db, thread_id: str, user_id: int, query: str, chunks_result: list, raw_items: list, metadata: dict):
    """
    儲存對話記錄到 agent_turns
    如果是新的 thread_id，也會在 agent_threads 建立記錄
    """
    # 檢查 thread_id 是否已存在於 agent_threads
    async with db.execute(
        "SELECT id FROM agent_threads WHERE thread_id = ?",
        (thread_id,)
    ) as cursor:
        thread_exists = await cursor.fetchone()

    if not thread_exists:
        # 建立新的 thread
        await db.execute(
            "INSERT INTO agent_threads (thread_id, user_id) VALUES (?, ?)",
            (thread_id, user_id)
        )
        print(f"Created new thread: {thread_id}")

    # 準備要儲存的資料
    raw_items_json = json.dumps(raw_items, ensure_ascii=False)
    output_json = json.dumps(chunks_result, ensure_ascii=False)
    metadata_json = json.dumps(metadata, ensure_ascii=False)

    # 插入新的 agent_turn
    await db.execute("""
        INSERT INTO agent_turns (thread_id, user_id, input, output, raw_items, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (thread_id, user_id, query, output_json, raw_items_json, metadata_json))

    await db.commit()
    print(f"Saved conversation to database for thread: {thread_id}")

def init_braintrust():
    global braintrust_logger, openai_client
    braintrust_logger = braintrust.init_logger(project=os.getenv("BRAINTRUST_PROJECT"))
    openai_client = braintrust.wrap_openai(AsyncOpenAI())
    set_default_openai_client(openai_client) # 設定 OpenAI Agents SDK 預設的 OpenAI Client 改用 braintrust 包裝過的版本
    return braintrust_logger, openai_client

tavily_client = None
def get_tavily_client():
    global tavily_client
    tavily_client = AsyncTavilyClient()
    return tavily_client

# Custom Agent Context
@dataclass
class CustomAgentContext:
    search_source: dict

# Function Tools
@function_tool
@braintrust.traced
async def knowledge_search(wrapper: RunContextWrapper[CustomAgentContext], query: str) -> str:
    """
    Search the web for information

    Args:
        query: The query keyword to search the web for.
    """

    print(f"  ⚙️ Calling knowledge_search with query: {query}")

    response = await get_tavily_client().search(query)

    wrapper.context.search_source[query] = response

    result = [ x["content"] for x in response["results"] ]

    print(f"  ⚙️ knowledge_search result: {result}")

    return str(result)

class GuardrailResult(BaseModel):
    allow: bool
    refusal_answer: str = Field(description="The reply to the user's question if allow is False, otherwise leave it blank.")

class ExtractFollowupQuestionsResult(BaseModel):
    followup_questions: list[str] = Field(description="3 follow-up questions exploring different aspects of the topic.")

# Agent Factory Functions
def create_guardrail_agent() -> Agent:
    """Create and return a guardrail agent instance"""
    return Agent(
        name="Guardrail Agent",
        instructions=load_prompt("guardrail"),
        model="gpt-5.4-mini",
        output_type=GuardrailResult,
    )

def create_followup_questions_agent() -> Agent:
    """Create and return a follow-up questions extraction agent instance"""
    return Agent(
        name="Extract Followup Questions Agent",
        instructions="""Your task is to extract 3 follow-up questions from the user's question and return them as "followup_questions": ["question1", "question2", "question3"].""",
        model="gpt-5.4-mini",
        output_type=ExtractFollowupQuestionsResult,
    )

def create_lead_agent() -> Agent[CustomAgentContext]:
    """Create and return a lead agent instance"""
    today = datetime.now().strftime("%Y-%m-%d")
    return Agent[CustomAgentContext](
        name="Lead Agent",
        instructions=load_prompt("lead"),
        tools=[knowledge_search],
        #tools=[
            #WebSearchTool(),
            #FileSearchTool(
            #  max_num_results=5,
            #  vector_store_ids=[os.getenv("OPENAI_VECTOR_STORE_ID")],
            #)
        #],
        model="gpt-5.6-luna",
        model_settings=ModelSettings(
            reasoning={
                "effort": "low",
                "summary": "auto"
            }
        )
    )

# Just for demo, not used in the actual system
@braintrust.traced
async def extract_conversation_metadata():
    await asyncio.sleep(2)

    return {
        "user_sentiment": "positive",
        "user_intent": "investment"
    }

@braintrust.traced
async def check_input_guardrail(input_items):
    input_guardrail_agent = create_guardrail_agent()

    result = await Runner.run(input_guardrail_agent, input=input_items)
    return result    