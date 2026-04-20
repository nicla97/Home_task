from typing import Any

from rag_task.models import Chunk, Document, Query, RetrievedChunk


def split_text_fixed(text: str, max_chars: int) -> list[str]:
    '''
    Split text into fixed-size chunks.

    Expected behavior:
    - strip surrounding whitespace
    - return [] for empty text
    - if max_chars <= 0, raise ValueError
    '''
    raise NotImplementedError


def chunk_document(document: Document, max_chars: int = 80) -> list[Chunk]:
    '''
    Convert a Document into retrieval Chunk objects.

    Rules:
    - preserve document and section metadata
    - skip empty sections
    - split long content using split_text_fixed
    - use deterministic chunk IDs
    '''
    raise NotImplementedError


def filter_chunks_by_metadata(
    chunks: list[RetrievedChunk],
    metadata_filter: dict[str, Any] | None,
) -> list[RetrievedChunk]:
    '''
    Return only chunks that match every key/value pair in metadata_filter.
    If metadata_filter is empty or None, return the input unchanged.
    '''
    raise NotImplementedError


def hybrid_retrieve(
    *,
    dense_results: list[RetrievedChunk],
    lexical_results: list[RetrievedChunk],
    query: Query,
    rank_constant: int = 60,
    solution_boost: float = 1.15,
) -> list[RetrievedChunk]:
    '''
    Fuse dense_results and lexical_results with Reciprocal Rank Fusion.

    Requirements:
    - apply query.metadata filtering before fusion
    - merge duplicates by chunk_id
    - keep document_id, text, section_type, page_number, metadata
    - assign the fused score to the returned RetrievedChunk.score
    - apply solution_boost after fusion
    - sort by score descending, then chunk_id ascending
    - return at most query.top_k items
    '''
    raise NotImplementedError


def build_answer_payload(
    query: Query,
    retrieved_chunks: list[RetrievedChunk],
    max_citation_chunks: int = 3,
) -> dict[str, Any]:
    '''
    Build a response payload with:
    - answer
    - citations
    - metadata

    Do not call any external model.
    '''
    raise NotImplementedError
