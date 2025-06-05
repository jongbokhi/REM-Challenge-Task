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
    cache_dir: str,
    db_index: str,
    texts: str | list[str] | None = None,
    model_name: str = "text-embedding-3-small",
    chunk_size: int = 1500,
    chunk_overlap: int = 150,
) -> FAISS:
    """Create or load a FAISS vector store.

    Parameters
    ----------
    cache_dir : str
        Directory to store embedding cache.
    db_index : str
        Directory to load or save the FAISS index.
    texts : str | list[str] | None, optional
        Input text(s) to index. When ``db_index`` already exists and this
        argument is ``None``, the function will attempt to read the text from
        ``db_index/raw_text.txt`` if present.
    model_name : str, optional
        Name of the OpenAI embedding model to use.
    chunk_size, chunk_overlap : int, optional
        Parameters for document chunking.

    Returns
    -------
    FAISS
        Loaded or newly created FAISS vector store.
    """

    embedder = get_cached_embedder(cache_dir, model_name)

    # ===== 1. Load existing index if present =====
    if os.path.isdir(db_index):
        db = FAISS.load_local(
            db_index,
            embedder,
            allow_dangerous_deserialization=True,
        )

        if texts is None:
            raw_file = os.path.join(db_index, "raw_text.txt")
            if os.path.isfile(raw_file):
                with open(raw_file, "r", encoding="utf-8") as f:
                    texts = f.read()

        return db

    # ===== 2. Build a new index =====
    if texts is None:
        raise ValueError("texts must be provided when creating a new index")

    os.makedirs(db_index, exist_ok=True)

    raw_file = os.path.join(db_index, "raw_text.txt")
    joined = "\n".join(texts) if isinstance(texts, list) else texts
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(joined)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_text(joined)

    faiss_db = FAISS.from_texts(split_docs, embedder)
    faiss_db.save_local(db_index)

    return FAISS.load_local(
        db_index,
        embedder,
        allow_dangerous_deserialization=True,
    )
