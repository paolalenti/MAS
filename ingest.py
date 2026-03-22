from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pathlib import Path


KB_PATH = "./knowledge_base"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "edu_docs"
DOC_TYPE_MAP = {
    "style": "style_guide",
    "docker": "docker_docs",
}


def detect_doc_type(file_path: str) -> str:
    """Определяет тип документа по пути к файлу."""
    path_lower = file_path.lower()
    for keyword, doc_type in DOC_TYPE_MAP.items():
        if keyword in path_lower:
            return doc_type
    return "general"


def run_ingestion():
    loader = DirectoryLoader(
        KB_PATH,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    docs = loader.load()

    for doc in docs:
        source_path = doc.metadata.get("source", "")
        doc.metadata["doc_type"] = detect_doc_type(source_path)
        doc.metadata["filename"] = Path(source_path).name

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

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
