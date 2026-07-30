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

# Serves static files locally
current_file_path = os.path.abspath(__file__)
api_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(api_directory)
public_directory = os.path.join(project_root, "public")

if os.path.exists(public_directory):
    app.mount("/", StaticFiles(directory=public_directory, html=True), name="public")