from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse, MemoryInjectRequest
from app.services.npc_service import NPCService
from app.services.chromadb_service import ChromaDbService

router = APIRouter()
npc_service = NPCService()
memory_service = ChromaDbService()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response_text, used_mem = npc_service.process_chat(request.npc_id, request.player_message)
        return ChatResponse(
            npc_id=request.npc_id,
            response_text=response_text,
            memory_used=used_mem
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/inject_memory")
async def inject_memory(request: MemoryInjectRequest):
    try:
        memory_service.add_memory(request.npc_id, request.fact, request.source)
        return {"status": "success", "message": "Pamięć zapisana w ChromaDB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    