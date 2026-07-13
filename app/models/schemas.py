from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    npc_id: str
    player_message: str
    context_radius: float = Field(default=5.0, description="Promień słyszenia/widzenia gracza")

class ChatResponse(BaseModel):
    npc_id: str
    response_text: str
    memory_used: bool

class MemoryInjectRequest(BaseModel):
    npc_id: str
    fact: str
    source: str = Field(default="observation", description="np. 'gossip', 'direct_interaction'")