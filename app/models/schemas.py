from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    npc_id: str
    player_message: str
    context_radius: float = Field(default=5.0, description="Radius in meters to consider for context retrieval. Default is 5.0 meters.")

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
