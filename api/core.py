import os
from phoenix.otel import register

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
PHOENIX_URL = os.getenv("PHOENIX_URL", "http://localhost:4317")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "edu_docs")

register(endpoint=PHOENIX_URL)

from openinference.instrumentation.langchain import LangChainInstrumentor

if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
    LangChainInstrumentor().instrument()

import json
from typing import Annotated, TypedDict, List, Any
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from qdrant_client import QdrantClient

from system_prompts import *

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

knowledge_retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.75}
)


class AgentState(TypedDict):
    topic: str
    course_plan: List[str]
    current_content: str
    quiz: dict
    user_score: int
    messages: Annotated[List[Any], add_messages]
    tool_call_count: int


llm = ChatOllama(
    # model="llama3.1:8b",  # Требует большего количества RAM
    # model="llama3.2:3b",  # Слишком глупая и не вызывает инструменты
    model="qwen2.5:7b",  # За 4m10s вернула удовлетворительный ответ
    # model="qwen3:4b",  # Не проверял
    base_url=OLLAMA_URL,
    temperature=0.7
)


def architect_node(state: AgentState):
    topic = state["topic"]

    docs = knowledge_retriever.invoke(f"{topic} style guide")
    context = "\n".join([d.page_content for d in docs])

    system_prompt = get_prompt(architect_prompt, topic=topic, context=context)

    response = llm.invoke(SystemMessage(content=system_prompt))
    plan = [p.strip() for p in response.content.split(",")]

    return {"course_plan": plan, "messages": [response]}


def content_creator_node(state: AgentState):
    plan = state["course_plan"]
    target_module = plan[0] if plan else state["topic"]

    instruments = "- 'search_knowledge_base' — используй когда нужны технические детали, команды, примеры"
    system_prompt = get_prompt(content_creator_prompt, module=target_module, instruments=instruments)

    all_messages = state["messages"]

    tool_dialogue = []

    for msg in all_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_dialogue = [msg]
        elif isinstance(msg, ToolMessage) and tool_dialogue:
            tool_dialogue.append(msg)

    messages = [SystemMessage(content=system_prompt)] + tool_dialogue

    response = llm_with_tools.invoke(messages)
    output = {"messages": [response]}

    if response.tool_calls:
        output["tool_call_count"] = state.get("tool_call_count", 0) + 1
    else:
        output["current_content"] = response.content
        output["tool_call_count"] = 0

    return output


def quiz_master_node(state: AgentState):
    plan = state["course_plan"]
    target_module = plan[0] if plan else state["topic"]
    content = state["current_content"]

    system_prompt = get_prompt(quiz_master_prompt, module=target_module, content=content)

    response = llm.invoke(SystemMessage(content=system_prompt))

    try:
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        quiz_json = json.loads(clean_content)
    except:
        quiz_json = {"error": "Не удалось создать валидный тест"}

    return {"quiz": quiz_json, "messages": [response]}


@tool
def search_knowledge_base(query: str) -> str:
    """Ищет технические факты по запросу или ключевым словам: команды, концепции, архитектуру, примеры использования."""
    docs = knowledge_retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


tools = [search_knowledge_base]
tool_node = ToolNode(tools)
llm_with_tools = llm.bind_tools(tools)


def router(state: AgentState):
    last_message = state["messages"][-1]
    if (
        hasattr(last_message, 'tool_calls')
        and last_message.tool_calls
        and state.get("tool_call_count", 0) < 3
    ):
        return "tools"
    return "quiz_master"


workflow = StateGraph(AgentState)

workflow.add_node("architect", architect_node)
workflow.add_node("content_creator", content_creator_node)
workflow.add_node("tools", tool_node)
workflow.add_node("quiz_master", quiz_master_node)

workflow.add_edge(START, "architect")
workflow.add_edge("architect", "content_creator")
workflow.add_conditional_edges("content_creator", router)
workflow.add_edge("tools", "content_creator")
workflow.add_edge("quiz_master", END)

app = workflow.compile()

if __name__ == "__main__":
    initial_input = {
        "topic": "Основы Docker для новичков",
        "messages": [],
        "course_plan": [],
        "current_content": "",
        "quiz": {},
        "user_score": 0,
        "tool_call_count": 0,
    }

    print("--- Запуск мультиагентной системы ---")
    result = app.invoke(initial_input)

    print("\n[Архитектор] составил план:")
    print(result["course_plan"])

    print("\n[Создатель] написал контент:")
    print(result["current_content"][:300] + "...")

    print("\n[Quiz Master] составил тест:")
    print(result["quiz"])