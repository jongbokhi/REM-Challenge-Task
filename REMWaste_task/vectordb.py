import os
from langchain.storage import LocalFileStore
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_cached_embedder(
    cache_dir: str,
    model_name: str = "text-embedding-3-small",
) -> CacheBackedEmbeddings:
    """
    Returns an embedding wrapper that supports caching based on LocalFileStore.
    """
    os.makedirs(cache_dir, exist_ok=True)
    store = LocalFileStore(cache_dir)
    base_embed = OpenAIEmbeddings(model=model_name)
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=base_embed,
        document_embedding_cache=store,
        namespace=base_embed.model,
    )


def build_vectordb(
    texts,
    cache_dir: str,
    db_index: str,
    model_name: str = "text-embedding-3-small",
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> FAISS:
    """
    1) texts: A single string or a list of strings
    2) cache_dir: Directory to store embedding cache
    3) db_index: Directory to load or save the FAISS index
    4) model_name: Name of the OpenAI embedding model to use
    5) chunk_size / chunk_overlap: Parameters for document chunking

    Returns: A FAISS vector store (either loaded or newly created)
    """
    # ============= 1. Split Text =============
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    split_docs = splitter.split_text(texts)

    # ============= 2. Embedding =============
    embedder = get_cached_embedder(cache_dir, model_name)

    # ============= 3.Check whether the index is loaded =============
    if os.path.isdir(db_index):
        db = FAISS.load_local(
            db_index,
            embedder,
            allow_dangerous_deserialization=True,
        )
        return db

    # ============= 4. Create and save the index=============
    faiss_db = FAISS.from_texts(split_docs, embedder)
    os.makedirs(db_index, exist_ok=True)
    faiss_db.save_local(db_index)

    db = FAISS.load_local(
        db_index,
        embedder,
        allow_dangerous_deserialization=True,
    )

    return db
