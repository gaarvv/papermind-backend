import chromadb
from chromadb.config import Settings
from services.embeddings import embed, embed_one


class VectorStore:
    def __init__(self):
        self._client = chromadb.EphemeralClient(
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.create_collection(
            name="papermind",
            metadata={"hnsw:space": "cosine"},
        )
        self._registry: dict[str, str] = {}  # doc_id → filename

    # ── write ────────────────────────────────────────────────────────────────

    def add_documents(self, chunks: list, doc_id: str, filename: str):
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = embed(texts)
        self._collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids,
            metadatas=metadatas,
        )
        self._registry[doc_id] = filename

    def delete_document(self, doc_id: str):
        results = self._collection.get(where={"doc_id": doc_id})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
        self._registry.pop(doc_id, None)

    def reset(self):
        self._client.delete_collection("papermind")
        self._collection = self._client.create_collection(
            name="papermind",
            metadata={"hnsw:space": "cosine"},
        )
        self._registry.clear()

    # ── read ─────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list:
        n = min(k, self._collection.count())
        if n == 0:
            return []
        qe = embed_one(query)
        results = self._collection.query(
            query_embeddings=[qe],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": round(1 - results["distances"][0][i], 4),
            })
        return chunks

    def list_documents(self) -> list:
        return [
            {"doc_id": did, "filename": fname}
            for did, fname in self._registry.items()
        ]

    def count(self) -> int:
        return self._collection.count()
