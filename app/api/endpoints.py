from fastapi import APIRouter, HTTPException
from loguru import logger
from app.models.schemas import ChatRequest, ChatResponse, MemoryInjectRequest
from app.services.npc_service import NPCService
from app.services.chromadb_service import ChromaDbService

router = APIRouter()
npc_service = NPCService()
memory_service = ChromaDbService()

@router.post("/chat", response_model=ChatResponse, tags=["Chat"])
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


@router.get("/npcIds", tags=["NPC profiles"])
async def get_all_npcs():
    try:
        npc_ids = npc_service.get_all_npc_ids()
        logger.info(f"Retrieved {len(npc_ids)} NPC IDs successfully.")
        return {
            "count": len(npc_ids),
            "npc_ids": npc_ids
        }
    
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/npcProfiles/{npc_id}", tags=["NPC profiles"])
async def get_npc(npc_id: str):
    try:
        profile = npc_service.get_npc_profile(npc_id)
        logger.info(f"Successfully retrieved profile for NPC ID: {npc_id}")
        return profile
    
    except FileNotFoundError:
        logger.warning(f"NPC profile for ID {npc_id} not found.")
        raise HTTPException(status_code=404, detail="NPC not found.")
    
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memories/{npc_id}", tags=["Memories"])
async def get_npc_memories(npc_id: str):
    try:
        memories = memory_service.get_all_memories(npc_id)
        logger.info(f"Retrieved {len(memories['memories'])} memories for NPC ID: {npc_id}")
        return memories
    
    except Exception as e:
        logger.error(f"Error occurred while retrieving memories for {npc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/injectMemory", tags=["Memories"])
async def inject_memory(request: MemoryInjectRequest):
    try:
        memory_service.add_memory(request.npc_id, request.fact, request.source)
        logger.info(f"Memory injected for {request.npc_id}. Fact: {request.fact}")
        return {
            "status": "success",
            "message": "Memory saved in ChromaDB."
        }
    
    except Exception as e:
        logger.error(f"Error occurred while injecting memory for {request.npc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/memories/{npc_id}", tags=["Memories"])
async def delete_all_memories(npc_id: str):
    try:
        deleted = memory_service.delete_all_memories(npc_id)
        logger.info(f"Deleted {deleted} memories for NPC {npc_id}")
        return {
            "status": "success",
            "npc_id": npc_id,
            "deleted_memories": deleted
        }
    
    except Exception as e:
        logger.error(f"Error deleting memories for {npc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/memories/{npc_id}/{memory_id}", tags=["Memories"])
async def delete_memory(npc_id: str, memory_id: str):
    try:
        deleted = memory_service.delete_memory(npc_id, memory_id)
        if not deleted:
            logger.warning(f"Memory {memory_id} not found for NPC {npc_id}")
            raise HTTPException(
                status_code=404,
                detail="Memory not found for specified NPC."
            )
        
        logger.info(f"Deleted memory {memory_id} for NPC {npc_id}")
        return {
            "status": "success",
            "npc_id": npc_id,
            "memory_id": memory_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    