from fastapi import APIRouter, HTTPException
from loguru import logger
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
        logger.info(f"Successfully generated response for {request.npc_id}. Memory used: {used_mem}")
        return ChatResponse(
            npc_id=request.npc_id,
            response_text=response_text,
            memory_used=used_mem
        )
    except FileNotFoundError as e:
        logger.warning(f"NPC with ID {request.npc_id} not found: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error occurred for {request.npc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

#note: Check endpoint construction guidelines for accessing resources in a RESTful manner.
@router.get("/memories/{npc_id}")
async def get_npc_memories(npc_id: str):
    try:
        memories = memory_service.get_all_memories(npc_id)
        logger.info(f"Retrieved {len(memories['memories'])} memories for NPC ID: {npc_id}")
        return memories
    except Exception as e:
        logger.error(f"Error occurred while retrieving memories for {npc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/injectMemory")
async def inject_memory(request: MemoryInjectRequest):
    try:
        memory_service.add_memory(request.npc_id, request.fact, request.source)
        logger.info(f"Memory injected for {request.npc_id}. Fact: {request.fact}")
        return {"status": "success", "message": "Memory saved in ChromaDB."}
    except Exception as e:
        logger.error(f"Error occurred while injecting memory for {request.npc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
#note: add more CRUD endpoints for memory management as needed, following RESTful principles.
    