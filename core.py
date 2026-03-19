import os
from phoenix.otel import register


os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:4317"

register()
from openinference.instrumentation.langchain import LangChainInstrumentor

if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
    LangChainInstrumentor().instrument()


from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "edu_docs"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})


class AgentState(TypedDict):
    topic: str
    messages: Annotated[List, add_messages]

    course_plan: List[str]

    current_content: str

    user_score: int

llm = ChatOllama(model="llama3.2:3b", temperature=0.7)
# llm = ChatOllama(model="qwen3:4b", temperature=0.7)


def architect_node(state: AgentState):
    topic = state["topic"]

    docs = retriever.invoke(f"{topic} style guide")
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""Ты — главный архитектор образовательных программ. 
    Используй эти правила и знания:
    {context}

    Задача: составь план из 3 модулей по теме: {topic}. 
    Отвечай СТРОГО списком через запятую, без лишних слов."""

    response = llm.invoke(prompt)
    plan = [p.strip() for p in response.content.split(",")]

    return {"course_plan": plan, "messages": [response]}


def content_creator_node(state: AgentState):
    plan = state["course_plan"]
    target_module = plan[0] if plan else "Введение"

    docs = retriever.invoke(target_module)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""Ты — создатель контента. Напиши подробный текст для модуля: {target_module}.
    Используй дополнительные знания:
    {context}

    Соблюдай стиль из базы знаний (эмодзи, обращение на 'ты')."""

    response = llm.invoke(prompt)
    return {"current_content": response.content, "messages": [response]}


workflow = StateGraph(AgentState)

workflow.add_node("architect", architect_node)
workflow.add_node("creator", content_creator_node)

workflow.add_edge(START, "architect")
workflow.add_edge("architect", "creator")
workflow.add_edge("creator", END)

app = workflow.compile()

if __name__ == "__main__":
    initial_input = {
        "topic": "Основы Docker для новичков",
        "messages": [],
        "course_plan": [],
        "current_content": "",
        "user_score": 0
    }

    print("--- Запуск мультиагентной системы ---")
    result = app.invoke(initial_input)

    print("\n[Архитектор] составил план:")
    print(result['course_plan'])

    print("\n[Создатель] написал контент для первого модуля:")
    print(result['current_content'][:300] + "...")
    print("-" * 30)
