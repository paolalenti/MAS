import os

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from pathlib import Path

KB_PATH = "./knowledge_base"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "edu_docs")


def get_title(filepath: str) -> str:
    try:
        line = open(filepath).readline().strip()
    except Exception:
        return ""
    return line[2:].strip() if line.startswith("# ") else ""


def run_ingestion():
    print("Загрузка чанков...")

    loader = DirectoryLoader(
        KB_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    for doc in docs:
        source_path = doc.metadata.get("source", "")
        doc.metadata["title"] = get_title(source_path)
        doc.metadata["filename"] = Path(source_path).name

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client = QdrantClient(url=QDRANT_URL)

    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if exists:
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Коллекция '{COLLECTION_NAME}' удалена.")
        else:
            print(f"Коллекция '{COLLECTION_NAME}' не найдена, создаём новую.")
    except Exception as e:
        print(f"Ошибка при проверке: {e}")
        raise

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )

    QdrantVectorStore.from_documents(
        splits,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    print(f"Готово! Загружено {len(splits)} чанков в векторную БД.")


if __name__ == "__main__":
    run_ingestion()
