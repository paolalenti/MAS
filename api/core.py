import os
import json

from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor
from typing import Annotated, TypedDict, List, Any, Dict
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_core.messages import SystemMessage
from qdrant_client import QdrantClient

from system_prompts import *

if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
    LangChainInstrumentor().instrument()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:4317")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "edu_docs")

register(endpoint=PHOENIX_URL)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

knowledge_retriever = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30}
)


class AgentState(TypedDict):
    topic: str
    course_plan: List[str]
    current_module_index: int
    current_content: str
    modules_content: Dict[str, str]
    quiz: dict
    all_quizzes: Dict[str, dict]
    user_score: int
    messages: Annotated[List[Any], add_messages]
    tool_call_count: int
    tool_messages_for_current_module: List[Any]


llm = ChatOllama(
    # model="qwen3.5:9b",
    model="qwen2.5:7b",
    base_url=OLLAMA_URL,
    temperature=0.7
)


def architect_node(state: AgentState):
    topic = state["topic"]

    docs = knowledge_retriever.invoke(topic)
    context = "\n".join([d.page_content for d in docs])

    system_prompt = get_prompt(architect_prompt, topic=topic, context=context)

    response = llm.invoke([SystemMessage(content=system_prompt)])
    plan = [p.strip() for p in response.content.split(",")]

    return {
        "course_plan": plan,
        "current_module_index": 0,
        "modules_content": {},
        "all_quizzes": {},
        "messages": [response],
    }


def content_creator_node(state: AgentState):
    plan = state["course_plan"]
    idx = state["current_module_index"]
    target_module = plan[idx]

    instruments = """ - 'search_knowledge_base' — 
    Вызывай этот инструмент ВСЕГДА и ПЕРВЫМ ДЕЛОМ, до написания любого текста.
    Используй для получения технических деталей по теме модуля: команды, конфигурации,
    примеры кода, архитектурные концепции, best practices.
    Аргумент query — ключевые слова или вопрос по теме модуля на русском языке.
    Пример: query="docker run команды и флаги"
    """

    system_prompt = get_prompt(content_creator_prompt, module=target_module, instruments=instruments)

    tool_dialogue = state.get("tool_messages_for_current_module", [])
    messages = [SystemMessage(content=system_prompt)] + tool_dialogue

    response = llm_with_tools.invoke(messages)
    output = {"messages": [response]}

    if response.tool_calls:
        output["tool_messages_for_current_module"] = tool_dialogue + [response]
        output["tool_call_count"] = state.get("tool_call_count", 0) + 1
    else:
        output["current_content"] = response.content
        output["tool_call_count"] = 0
        output["tool_messages_for_current_module"] = []

    return output


def quiz_master_node(state: AgentState):
    plan = state["course_plan"]
    idx = state["current_module_index"]
    target_module = plan[idx]
    content = state["current_content"]

    system_prompt = get_prompt(quiz_master_prompt, module=target_module, content=content)

    response = llm.invoke([SystemMessage(content=system_prompt)])

    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        quiz_json = json.loads(clean_content)
    except Exception:
        quiz_json = {"error": f"Не удалось создать валидный тест для модуля «{target_module}»"}

    modules_content = dict(state.get("modules_content", {}))
    modules_content[target_module] = content

    all_quizzes = dict(state.get("all_quizzes", {}))
    all_quizzes[target_module] = quiz_json

    return {
        "quiz": quiz_json,
        "all_quizzes": all_quizzes,
        "modules_content": modules_content,
        "messages": [response],
    }


def advance_module_node(state: AgentState):
    """новый узел — переходим к следующему модулю"""
    return {
        "current_module_index": state["current_module_index"] + 1,
        "current_content": "",
        "tool_messages_for_current_module": [],
        "tool_call_count": 0,
    }


@tool
def search_knowledge_base(query: str) -> str:
    """Ищет технические факты по запросу или ключевым словам: команды, концепции, архитектуру, примеры использования"""
    docs = knowledge_retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


tools = [search_knowledge_base]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)


def content_router(state: AgentState):
    """Роутер после content_creator: идём в tools или в quiz_master"""
    last_message = state["messages"][-1]
    if (
            hasattr(last_message, "tool_calls")
            and last_message.tool_calls
            and state.get("tool_call_count", 0) < 3
    ):
        return "tools"
    return "quiz_master"


def tools_node_with_context(state: AgentState):
    """обёртка вокруг ToolNode — добавляем ToolMessage в изолированный диалог модуля"""
    result = tool_node.invoke(state)
    tool_msgs = result.get("messages", [])
    updated_dialogue = state.get("tool_messages_for_current_module", []) + tool_msgs
    return {**result, "tool_messages_for_current_module": updated_dialogue}


def module_loop_router(state: AgentState):
    """после quiz_master проверяем — есть ли ещё модули"""
    idx = state["current_module_index"]
    plan = state["course_plan"]
    if idx + 1 < len(plan):
        return "advance_module"
    return END


workflow = StateGraph(AgentState)

workflow.add_node("architect", architect_node)
workflow.add_node("content_creator", content_creator_node)
workflow.add_node("tools", tools_node_with_context)
workflow.add_node("quiz_master", quiz_master_node)
workflow.add_node("advance_module", advance_module_node)

workflow.add_edge(START, "architect")
workflow.add_edge("architect", "content_creator")
workflow.add_conditional_edges("content_creator", content_router)
workflow.add_edge("tools", "content_creator")
workflow.add_conditional_edges("quiz_master", module_loop_router, {"advance_module": "advance_module", END: END})
workflow.add_edge("advance_module", "content_creator")

app = workflow.compile()

if __name__ == "__main__":
    initial_input = {
        "topic": "Основы Docker для новичков",
        "messages": [],
        "course_plan": [],
        "current_module_index": 0,
        "current_content": "",
        "modules_content": {},
        "quiz": {},
        "all_quizzes": {},
        "user_score": 0,
        "tool_call_count": 0,
        "tool_messages_for_current_module": [],
    }

    print("--- Запуск мультиагентной системы ---")
    result = app.invoke(initial_input)

    print("\n[Архитектор] составил план:")
    print(result["course_plan"])

    print("\n[Создатель] написал контент по каждому модулю:")
    for module, content in result["modules_content"].items():
        print(f"\n── {module} ──")
        print(content[:300] + "...")

    print("\n[Quiz Master] составил тесты по каждому модулю:")
    for module, quiz in result["all_quizzes"].items():
        print(f"\n── {module} ──")
        print(quiz)
