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
    
    def get_all_memories(self, npc_id: str) -> dict:
        """Fetches all memories for a given NPC"""
        results = self.collection.get(
            where={"npc_id": npc_id}
        )
        return {
            "npc_id": npc_id,
            "total_memories": len(results['documents']),
            "memories": [
                {"id": m_id, "content": doc, "metadata": meta}
                for m_id, doc, meta in zip(results['ids'], results['documents'], results['metadatas'])
            ]
        }
    
    def delete_all_memories(self, npc_id: str) -> int:
        """
        Deletes all memories belonging to an NPC.
        Returns number of deleted memories.
        """
        results = self.collection.get(where={"npc_id": npc_id})
        ids = results.get("ids", [])
        if not ids:
            return 0
        
        self.collection.delete(ids=ids)
        return len(ids)
    
    def delete_memory(self, npc_id: str, memory_id: str) -> bool:
        """
        Deletes a single memory if it belongs to the specified NPC.
        Returns True if deleted.
        """
        results = self.collection.get(
            ids=[memory_id],
            include=["metadatas"]
        )
        if not results["ids"]:
            return False
        
        metadata = results["metadatas"][0]
        if metadata["npc_id"] != npc_id:
            return False

        self.collection.delete(ids=[memory_id])
        return True