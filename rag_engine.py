import math
import re
import os
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits document text into overlapping text strings to prepare for search indexing.
    """
    if not text:
        return []
    
    chunks = []
    text_len = len(text)
    
    # If text is smaller than chunk size, return it as a single chunk
    if text_len <= chunk_size:
        return [text]
        
    start = 0
    step = chunk_size - overlap
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break
        
        chunks.append(text[start:end])
        start += step
        
    return chunks

def tokenize(text: str) -> List[str]:
    """
    Splits text into lowercase words, ignoring punctuation marks.
    """
    return re.findall(r'\b\w+\b', text.lower())

class SimpleTFIDF:
    """
    A lightweight, zero-dependency TF-IDF vectorizer for offline document retrieval.
    """
    def __init__(self, documents: List[str]):
        """
        Builds the word vocabulary and pre-calculates TF-IDF vectors for all documents.
        """
        self.documents = documents
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.num_docs = len(documents)
        
        # Calculate vocabulary and document frequencies
        self.vocab = set()
        self.df = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            self.vocab.update(unique_tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
                
        # Calculate smoothed IDF
        self.idf = {}
        for token in self.vocab:
            self.idf[token] = math.log((1 + self.num_docs) / (1 + self.df[token])) + 1.0
            
        # Pre-calculate TF-IDF vectors for documents
        self.doc_vectors = []
        for tokens in self.doc_tokens:
            vec = self._vectorize(tokens)
            self.doc_vectors.append(vec)
            
    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        """
        Creates a dictionary of word term frequency multiplied by its inverse document frequency.
        """
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
            
        tf_idf = {}
        for t, count in tf.items():
            if t in self.idf:
                tf_idf[t] = count * self.idf[t]
        return tf_idf
        
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Measures the similarity of word usage between two sparse TF-IDF dictionaries.
        """
        dot = 0.0
        for k, v in vec1.items():
            if k in vec2:
                dot += v * vec2[k]
                
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)
        
    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Scores the search query against all documents and returns the top-k indices.
        """
        query_tokens = tokenize(query)
        query_vec = self._vectorize(query_tokens)
        
        scores = []
        for idx, doc_vec in enumerate(self.doc_vectors):
            sim = self._cosine_similarity(query_vec, doc_vec)
            scores.append((idx, sim))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

_SELECTED_EMBED_MODEL = None

def _get_embed_model() -> str:
    """
    Selects the best available vector embedding model from the active API key.
    """
    global _SELECTED_EMBED_MODEL
    if _SELECTED_EMBED_MODEL:
        return _SELECTED_EMBED_MODEL
        
    default_model = "models/gemini-embedding-2"
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return default_model
        
    try:
        genai.configure(api_key=api_key)
        models = [m.name for m in genai.list_models()]
        
        priorities = [
            "models/gemini-embedding-2",
            "models/text-embedding-004",
            "models/gemini-embedding-001"
        ]
        
        for p in priorities:
            if p in models:
                _SELECTED_EMBED_MODEL = p
                return _SELECTED_EMBED_MODEL
                
        for m in models:
            if "embedding" in m:
                _SELECTED_EMBED_MODEL = m
                return _SELECTED_EMBED_MODEL
    except Exception as e:
        print(f"[RAG Engine] Error listing models: {e}")
        
    _SELECTED_EMBED_MODEL = default_model
    return _SELECTED_EMBED_MODEL

class DocumentRAG:
    """
    Orchestrates chunking, Gemini embeddings calculation, and vector/TF-IDF similarity searches.
    """
    def __init__(self, text: str, chunk_size: int = 1000, overlap: int = 200):
        """
        Initializes the text document, chunks it, and sets up index variables.
        """
        self.text = text
        self.chunks = chunk_text(text, chunk_size, overlap)
        self.embeddings: Optional[List[List[float]]] = None
        self.tfidf: Optional[SimpleTFIDF] = None
        self.use_api = False
        
    def initialize(self, api_key: Optional[str] = None) -> None:
        """
        Retrieves vector embeddings online from Gemini or builds a local offline TF-IDF index.
        """
        if not self.chunks:
            return
            
        effective_key = api_key or os.environ.get("GEMINI_API_KEY")
        if effective_key:
            try:
                genai.configure(api_key=effective_key)
                emb_model = _get_embed_model()
                # Compute embeddings in a batch request
                result = genai.embed_content(
                    model=emb_model,
                    content=self.chunks,
                    task_type="retrieval_document"
                )
                self.embeddings = result.get("embedding")
                if self.embeddings:
                    self.use_api = True
                    return
            except Exception as e:
                # Log or print error, fall back to offline TF-IDF
                print(f"[RAG Engine] Embedding API failed, falling back to TF-IDF. Error: {e}")
                
        # Fallback to TF-IDF if API is unavailable or failed
        self.tfidf = SimpleTFIDF(self.chunks)
        self.use_api = False

    def _cosine_similarity_emb(self, a: List[float], b: List[float]) -> float:
        """
        Calculates the similarity between two floating-point embedding arrays.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x**2 for x in a))
        norm_b = math.sqrt(sum(y**2 for y in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Searches the vector index or offline TF-IDF fallback to return the top matching text chunks.
        """
        if not self.chunks:
            return []
            
        if self.use_api and self.embeddings:
            try:
                # Get embedding for the query
                emb_model = _get_embed_model()
                result = genai.embed_content(
                    model=emb_model,
                    content=query,
                    task_type="retrieval_query"
                )
                query_emb = result.get("embedding")
                if query_emb:
                    scores = []
                    for idx, chunk_emb in enumerate(self.embeddings):
                        sim = self._cosine_similarity_emb(query_emb, chunk_emb)
                        scores.append((idx, sim))
                    scores.sort(key=lambda x: x[1], reverse=True)
                    return [self.chunks[idx] for idx, _ in scores[:top_k]]
            except Exception as e:
                # Graceful fallback to TF-IDF on error
                print(f"[RAG Engine] Query embedding failed, using TF-IDF. Error: {e}")
                if not self.tfidf:
                    self.tfidf = SimpleTFIDF(self.chunks)
                
        # TF-IDF search fallback
        if not self.tfidf:
            self.tfidf = SimpleTFIDF(self.chunks)
        results = self.tfidf.search(query, top_k)
        return [self.chunks[idx] for idx, _ in results]
