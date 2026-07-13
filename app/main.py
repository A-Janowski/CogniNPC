from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(router, prefix="/api")

@app.get("/health")
async def root():
    return {"message": "CogniNPC Backend działa poprawnie."}