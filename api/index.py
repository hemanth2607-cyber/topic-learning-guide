import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from api.engine import GrokEngine

app = FastAPI()

class GenerationRequest(BaseModel):
    topic: str
    level: str

def get_engine():
    try:
        return GrokEngine()
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate(request: GenerationRequest):
    engine = get_engine()
    try:
        result = engine.generate_path(request.topic, request.level)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount the static files so local development ('run.py') serves index.html correctly
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")