import os

class Settings:
    PROJECT_NAME: str = "CogniNPC Core API"
    OLLAMA_URL: str = "http://172.25.16.1:11434/api/generate" 
    LLM_MODEL_LLama_3_1: str = "llama3.1:latest"
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CHROMA_DIR = os.path.join(DATA_DIR, "chromadb_data")
    PROFILES_DIR = os.path.join(DATA_DIR, "npc_profiles")

settings = Settings()