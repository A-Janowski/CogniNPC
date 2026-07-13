import chromadb
from app.core.config import settings

class ChromaDbService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(name="npc_memories")

    def add_memory(self, npc_id: str, fact: str, source: str):
        import uuid
        mem_id = str(uuid.uuid4())
        self.collection.add(
            documents=[fact],
            metadatas=[{"npc_id": npc_id, "source": source}],
            ids=[mem_id]
        )

    def retrieve_context(self, npc_id: str, query: str, limit: int = 2) -> str:
        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where={"npc_id": npc_id}
        )
        if results['documents'] and results['documents'][0]:
            return " ".join(results['documents'][0])
        return ""