"""Retrieval-Augmented Generation (RAG) solution."""

from typing import List, Optional, Tuple


class RAG:
    """Retrieval-Augmented Generation system.

    Combines a vector store for document retrieval with an LLM for generation.

    Args:
        model_id: HuggingFace model ID for generation.
        embedding_model: Sentence-transformers model for embeddings.
        device: Device for inference.
        max_tokens: Maximum tokens in generated answer.
        top_k: Number of documents to retrieve.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        max_tokens: int = 512,
        top_k: int = 3,
    ):
        self.model_id = model_id
        self.embedding_model_name = embedding_model
        self.device = device
        self.max_tokens = max_tokens
        self.top_k = top_k
        self._predictor = None
        self._embedder = None
        self._index = None
        self._documents: List[str] = []

    @property
    def predictor(self):
        if self._predictor is None:
            from flashllm.engine.predictor import Predictor

            self._predictor = Predictor(model_id=self.model_id, device=self.device)
        return self._predictor

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self.embedding_model_name, device=self.device)
            except ImportError:
                raise ImportError("RAG requires: pip install 'flashllm[rag]'")
        return self._embedder

    def add_documents(self, documents: List[str]):
        """Add documents to the vector store."""
        import numpy as np

        try:
            import faiss
        except ImportError:
            raise ImportError("RAG requires: pip install 'flashllm[rag]'")

        embeddings = self.embedder.encode(documents, convert_to_numpy=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._index.add(embeddings)
        self._documents.extend(documents)

    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """Query the RAG system with a question."""
        k = top_k or self.top_k
        retrieved = self.retrieve(question, top_k=k)
        context = "\n\n".join([f"Document {i + 1}:\n{doc}" for i, (doc, _) in enumerate(retrieved)])
        prompt = f"Answer using the context.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return self.predictor.generate(prompt, max_tokens=self.max_tokens, temperature=0.3)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Retrieve the most relevant documents for a query."""
        import numpy as np
        import faiss

        if self._index is None or not self._documents:
            return []
        query_emb = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)
        scores, indices = self._index.search(query_emb, min(top_k, len(self._documents)))
        return [(self._documents[idx], float(score)) for score, idx in zip(scores[0], indices[0]) if idx >= 0]

    def reset(self):
        """Clear all documents and the index."""
        self._index = None
        self._documents = []
