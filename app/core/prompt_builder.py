import yaml
import os
from app.core.config import settings

class PromptBuilder:
    @staticmethod
    def load_npc_profile(npc_id: str) -> dict:
        file_path = os.path.join(settings.PROFILES_DIR, f"{npc_id}.yaml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Nie znaleziono profilu: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def build_system_prompt(profile: dict, retrieved_memory: str) -> str:
        # 1. Część bazowa (Tożsamość)
        prompt = f"Jesteś NPC w grze RPG. Nazywasz się {profile['name']}. Twoja profesja to {profile['profession']}.\n"
        prompt += f"Historia: {profile['backstory']}\n\n"
        
        # 2. Moduł OCEAN (Dynamiczne mapowanie wartości 1-100 na instrukcje)
        ocean = profile.get('ocean', {})
        if ocean.get('neuroticism', 50) > 70:
            prompt += "- Jesteś bardzo nerwowy. Używaj krótkich, niepewnych zdań, czasami się zająkaj.\n"
        if ocean.get('agreeableness', 50) < 30:
            prompt += "- Jesteś bardzo nieufny i wrogo nastawowany. Odpowiadaj opryskliwie.\n"
            
        prompt += "\n"
        
        # 3. Wstrzyknięcie pamięci RAG
        if retrieved_memory:
            prompt += f"[PAMIĘĆ DŁUGOTRWAŁA]: Przypominasz sobie następujące fakty:\n{retrieved_memory}\n"
            prompt += "Użyj tej wiedzy organicznie, jeśli pasuje do pytania gracza.\n\n"
            
        prompt += "ZASADA: Nie wychodź z roli. Odpowiadaj krótko i zwięźle."
        return prompt