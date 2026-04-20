from rag_task.models import (
    Chunk,
    Document,
    DocumentSection,
    Query,
    dense_results,
    lexical_results,
    sample_documents,
)
from rag_task.task import (
    build_answer_payload,
    chunk_document,
    filter_chunks_by_metadata,
    hybrid_retrieve,
    split_text_fixed,
)


def test_split_text_fixed_and_chunk_document():
    chunks = chunk_document(sample_documents[0], max_chars=40)

    assert chunks, "Expected at least one chunk"
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.document_id == "KB-001" for chunk in chunks)
    assert all(chunk.metadata["source_type"] == "kb" for chunk in chunks)
    assert all(chunk.metadata["source_ref"] == "KB-001" for chunk in chunks)
    assert any(chunk.section_type == "solution" for chunk in chunks)

    solution_chunks = [chunk for chunk in chunks if chunk.section_type == "solution"]
    assert len(solution_chunks) >= 2, "Expected the solution section to be split into multiple chunks"

    empty_doc = Document(
        document_id="KB-EMPTY",
        source_type="kb",
        source_ref="KB-EMPTY",
        title="Empty doc",
        sections=[
            DocumentSection("description", "   "),
            DocumentSection("solution", ""),
        ],
    )

    assert chunk_document(empty_doc, max_chars=20) == []

    try:
        split_text_fixed("hello", 0)
        raise AssertionError("Expected ValueError when max_chars <= 0")
    except ValueError:
        pass


def test_hybrid_retrieve_and_filtering():
    query = Query(text="How do I reset VPN after password change?", top_k=3, metadata={"product": "vpn"})
    results = hybrid_retrieve(
        dense_results=dense_results,
        lexical_results=lexical_results,
        query=query,
        rank_constant=60,
        solution_boost=1.10,
    )

    assert len(results) == 2, "Only vpn chunks should remain after metadata filtering"
    assert results[0].chunk_id == "KB-001:solution:0", "Solution chunk should rank first after boost"
    assert results[1].chunk_id == "PDF-001:page_text:1"
    assert results[0].score > results[1].score

    no_filter_results = hybrid_retrieve(
        dense_results=dense_results,
        lexical_results=lexical_results,
        query=Query(text="install office", top_k=5),
    )

    assert len(no_filter_results) == 3, "Duplicate chunk_ids should be merged"
    assert len({chunk.chunk_id for chunk in no_filter_results}) == 3

    filtered = filter_chunks_by_metadata(dense_results, {"product": "vpn"})
    assert len(filtered) == 2
    assert all(chunk.metadata["product"] == "vpn" for chunk in filtered)


def test_build_answer_payload():
    ranked_results = hybrid_retrieve(
        dense_results=dense_results,
        lexical_results=lexical_results,
        query=Query(text="vpn reset", top_k=5, metadata={"product": "vpn"}),
    )

    payload = build_answer_payload(Query(text="vpn reset", top_k=5), ranked_results, max_citation_chunks=2)

    assert isinstance(payload, dict)
    assert set(payload.keys()) == {"answer", "citations", "metadata"}
    assert payload["metadata"]["result_count"] == len(ranked_results)
    assert payload["metadata"]["query"] == "vpn reset"
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["chunk_id"] == ranked_results[0].chunk_id
    assert "source_type" in payload["citations"][0]
    assert "source_ref" in payload["citations"][0]
    assert payload["answer"], "Expected a non-empty answer"

    empty_payload = build_answer_payload(Query(text="unknown"), [])
    assert empty_payload["answer"] == "I could not find relevant support content."
    assert empty_payload["citations"] == []
