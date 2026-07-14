from app.services.chromadb_service import ChromaDbService
from app.services.ollama_service import OllamaService
from app.core.prompt_builder import PromptBuilder
from loguru import logger

class NPCService:
    def __init__(self):
        self.memory = ChromaDbService()
        self.llm = OllamaService()
        self.prompt_builder = PromptBuilder()

    def process_chat(self, npc_id: str, player_message: str) -> tuple[str, bool]:
        # 1. Ładowanie YAML
        logger.debug(f"Loading NPC profile for ID: {npc_id}")
        profile = self.prompt_builder.load_npc_profile(npc_id)
        
        # 2. Wyszukiwanie w wektorowej pamięci (RAG)
        logger.debug(f"Retrieving context for NPC ID: {npc_id} with player message: {player_message}")
        context = self.memory.retrieve_context(npc_id, player_message)
        
        # 3. Budowa promptu
        logger.debug(f"Building system prompt for NPC ID: {npc_id}")
        system_prompt = self.prompt_builder.build_system_prompt(profile, context)
        full_prompt = f"{system_prompt}\n\nPlayer: {player_message}\nYou:"
        
        # 4. Inferencja
        logger.debug(f"Generating response for NPC ID: {npc_id}")
        response_text = self.llm.generate(full_prompt)
        
        # 5. (Opcjonalnie) Zapisz aktualną konwersację do pamięci na przyszłość
        logger.debug(f"Saving conversation for NPC ID: {npc_id}")
        self.memory.add_memory(npc_id, f"Player said: {player_message}. I responded: {response_text}", "conversation")

        used_memory = len(context) > 0
        logger.info(f"Successfully processed chat for {npc_id}. Memory used: {used_memory}")
        return response_text, used_memory