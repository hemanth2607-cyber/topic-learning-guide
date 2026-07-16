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

# Find the absolute path to the 'public' folder
# This works both locally and inside Vercel's serverless environment
current_file_path = os.path.abspath(__file__)        # api/index.py
api_directory = os.path.dirname(current_file_path)   # api/
project_root = os.path.dirname(api_directory)        # project root/
public_directory = os.path.join(project_root, "public")

if os.path.exists(public_directory):
    app.mount("/", StaticFiles(directory=public_directory, html=True), name="public")
else:
    print(f"Warning: Public directory not found at {public_directory}")