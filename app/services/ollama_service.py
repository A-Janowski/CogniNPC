import requests
from app.core.config import settings

class OllamaService:
    @staticmethod
    def generate(prompt: str) -> str:
        payload = {
            "model": settings.LLM_MODEL_LLama_3_1,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7, # Możemy to w przyszłości modyfikować dla szalonych NPC
                "num_ctx": 2048 # Ograniczenie okna kontekstowego dla oszczędności VRAM
            }
        }
        response = requests.post(settings.OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "").strip()