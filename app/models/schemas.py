from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    npc_id: str
    player_message: str
 
    # --- benchmark fields (optional) ---
    model: str | None = Field(
        default=None,
        description="Nadpisuje domyślny model Ollamy dla tego żądania."
    )
    rag_k: int | None = Field(
        default=None, ge=0,
        description="Liczba wektorów pobieranych z ChromaDB. None = wartość domyślna serwisu."
    )
    num_predict: int | None = Field(
        default=None, ge=1,
        description="Maksymalna liczba generowanych tokenów. Kontroluje długość wypowiedzi "
                    "tak, aby porównania między modelami/warunkami były znormalizowane."
    )
    skip_memory_write: bool = Field(
        default=False,
        description="Gdy True, pomija zapis konwersacji do ChromaDB. Domyślnie False, "
                    "więc zachowanie dla klienta Unity jest niezmienione. Benchmark "
                    "ustawia True, żeby retrieval przez cały przebieg operował na tej "
                    "samej, kontrolowanej puli wspomnień NPC — bez tego pula rośnie "
                    "monotonicznie w trakcie serii i wyniki dla rag_k=20 na końcu "
                    "przebiegu przestają być porównywalne z rag_k=20 na początku."
    )
class ChatResponse(BaseModel):
    npc_id: str
    response_text: str
    memory_used: bool
    metrics: dict = Field(default_factory=dict, description="Metrics from the LLM response, including load_duration, prompt_eval_duration, eval_duration, and eval_count")

class MemoryInjectRequest(BaseModel):
    npc_id: str
    fact: str
    source: str = Field(default="observation", description="source of the memory, e.g., 'observation', 'gossip', etc.")

class GossipRequest(BaseModel):
    npc_source: str
    npc_target: str
