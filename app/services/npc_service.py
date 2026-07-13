from app.services.chromadb_service import ChromaDbService
from app.services.ollama_service import OllamaService
from app.core.prompt_builder import PromptBuilder

class NPCService:
    def __init__(self):
        self.memory = ChromaDbService()
        self.llm = OllamaService()
        self.prompt_builder = PromptBuilder()

    def process_chat(self, npc_id: str, player_message: str) -> tuple[str, bool]:
        # 1. Ładowanie YAML
        profile = self.prompt_builder.load_npc_profile(npc_id)
        
        # 2. Wyszukiwanie w wektorowej pamięci (RAG)
        context = self.memory.retrieve_context(npc_id, player_message)
        
        # 3. Budowa promptu
        system_prompt = self.prompt_builder.build_system_prompt(profile, context)
        full_prompt = f"{system_prompt}\n\nGracz: {player_message}\nTy:"
        
        # 4. Inferencja
        response_text = self.llm.generate(full_prompt)
        
        # 5. (Opcjonalnie) Zapisz aktualną konwersację do pamięci na przyszłość
        # self.memory.add_memory(npc_id, f"Gracz powiedział: {player_message}. Odpowiedziałem: {response_text}", "conversation")

        used_memory = len(context) > 0
        return response_text, used_memory