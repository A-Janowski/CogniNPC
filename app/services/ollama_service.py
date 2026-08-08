import requests
from app.core.config import settings

class OllamaService:
    @staticmethod
    def generate(prompt: str) -> dict:
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
            data = response.json()

            return {
                "response_text": data.get("response", "").strip(),           # generated text response from the model
                "load_duration": data.get("load_duration", 0),               # time to load the model into VRAM (nanoseconds)
                "prompt_eval_duration": data.get("prompt_eval_duration", 0), # time to read the prompt (nanoseconds)
                "eval_duration": data.get("eval_duration", 0),               # time to generate the response (nanoseconds)
                "eval_count": data.get("eval_count", 0)                      # number of tokens evaluated (prompt + response)
            }
        except requests.exceptions.Timeout:
            return {"response_text": "[Timeout Error]", "eval_count": 0}
        except requests.exceptions.RequestException as e:
            return {"response_text": f"[Error: {str(e)}]", "eval_count": 0}