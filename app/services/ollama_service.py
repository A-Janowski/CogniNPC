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
                "temperature": 0.7,
                "num_ctx": 2048
            }
        }
        try:
            response = requests.post(settings.OLLAMA_URL, json=payload, timeout=60.0)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            return "System error: The request to the LLM engine timed out. Please try again later."
        except requests.exceptions.RequestException as e:
            return f"System error: Error communicating with the LLM engine: {str(e)}"