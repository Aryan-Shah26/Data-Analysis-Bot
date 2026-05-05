import logging
import numpy as np
from config import (
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
    RAG_TOP_K,
    EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

_embedder = None

def _get_embedder() :
    """
    Loads the sentence transformer model once and resuses it.
    Lazy loading to avoid slow down at startup.
    """

    global _embedder
    if _embedder is None :
        logger.info("Loading embedding model...")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _embedder

def _chunk_text(text : str) -> list[str] :
    """
    Splits text into overlapping chunks for RAG.
    """
    chunks = []
    start = 0

    while start < len(text) :
        end = start + RAG_CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        start += RAG_CHUNK_SIZE - RAG_CHUNK_OVERLAP

    return chunks

def build_rag_index(text : str) ->tuple :
    """
    Takes raw domain context, chunks and embeds it, and builds a FAISS index for similarity search.
    
    Returns:
        index: FAISS index
        chunks: list of raw text chunks (for retrieval)
    """
    import faiss

    if not text.strip() :
        return None, []

    chunks = _chunk_text(text)
    if not chunks :
        return None, []
    
    logger.info(f"Building RAG index from {len(chunks)} chunks...")

    embedder = _get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    logger.info(f"RAG index built: {index.ntotal} vectors, dimension {dimension}.")
    return index, chunks

def retrieve_context(query : str, index, chunks : list, top_k = RAG_TOP_K) -> str :
    """
    Retrieves the most relevant context chunks for a given query.
    Returns a single concatenated string of the top K results.
    """
    import faiss

    if index is None or not chunks :
        return ""
    
    embedder = _get_embedder()
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, min(top_k, len(chunks)))
    relevant_chunks = [
        chunks[i]
        for score,i in zip(scores[0], indices[0])
        if score > 0.3 and i < len(chunks)
    ]

    return "\n\n".join(relevant_chunks)

def retrieve_context_for_columns(columns : list[str], index, chunks : list) -> dict :
    """
    Retrieves domain context for eachh column in one call.
    Returns dict {"column_name" : context_str}
    """

    if index is None or not chunks :
        return {}
    
    context_map ={}
    for col in columns :
        query = f"column {col} description valid range constraints meaning"
        context = retrieve_context(query, index, chunks, RAG_TOP_K)
        if context :
            context_map[col] = context
    return context_map
