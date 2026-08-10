import requests
from app.core.config import settings

class OllamaService:
    @staticmethod
    def generate(prompt: str, model: str | None = None, num_predict: int | None = None) -> dict:
        payload = {
            "model": model or settings.LLM_MODEL_LLama_3_1,
            "prompt": prompt,
            "stream": False,
            # keep_alive="30m" zapobiega wyładowaniu wag z VRAM między
            # żądaniami benchmarku. Bez tego pojedyncze przeładowania
            # (load_duration rzędu sekund) losowo zaburzają serię pomiarów —
            # backend nie odróżnia wtedy "zimnego startu" od stanu ustalonego.
            "keep_alive": "30m",
            "options": {
                "temperature": 0.7,
                "num_ctx": 2048,
                **({"num_predict": num_predict} if num_predict is not None else {}),
            },
        }
        try:
            response = requests.post(settings.OLLAMA_URL, json=payload, timeout=300.0)
            response.raise_for_status()
            data = response.json()
 
            return {
                "response_text": data.get("response", "").strip(),
                "load_duration": data.get("load_duration", 0),
                "prompt_eval_duration": data.get("prompt_eval_duration", 0),
                "eval_duration": data.get("eval_duration", 0),
                "eval_count": data.get("eval_count", 0),
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "total_duration": data.get("total_duration", 0),
                "model_used": payload["model"],
            }
        except requests.exceptions.Timeout:
            return {"response_text": "[Timeout Error]", "eval_count": 0,
                    "prompt_eval_count": 0, "error": "timeout"}
        except requests.exceptions.RequestException as e:
            return {"response_text": f"[Error: {str(e)}]", "eval_count": 0,
                    "prompt_eval_count": 0, "error": str(e)}