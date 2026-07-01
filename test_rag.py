from rag_engine import chunk_text, DocumentRAG

def test_chunking_short_text():
    text = "This is a short text."
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    assert chunks[0] == text

def test_chunking_long_text():
    text = "A" * 1500
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 2
    # Ensure chunk size is respected
    assert len(chunks[0]) == 1000
    # The remainder is 500, with overlap we start from 800 (1000 - 200).
    # So the second chunk is from index 800 to 1500, which has length 700.
    assert len(chunks[1]) == 700

def test_tfidf_search():
    doc_text = (
        "This is paragraph number one which discusses apples and oranges.\n\n"
        "Here is paragraph number two which talks about computers and coding.\n\n"
        "Finally, paragraph number three focuses on space exploration and planets."
    )
    # Use chunk size that splits them into distinct chunks
    rag = DocumentRAG(doc_text, chunk_size=100, overlap=10)
    rag.initialize(api_key=None) # Forces TF-IDF
    
    # Query for apples should return paragraph one
    results_apples = rag.search("apples", top_k=1)
    assert len(results_apples) == 1
    assert "apples" in results_apples[0]
    
    # Query for coding should return paragraph two
    results_coding = rag.search("coding", top_k=1)
    assert len(results_coding) == 1
    assert "coding" in results_coding[0]
