from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import os
from typing import Dict, Optional
import asyncio
import logging
from processor import process_youtube_url
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Podcast Review API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs(os.path.join("public", "audio"), exist_ok=True)
os.makedirs("output", exist_ok=True)

# Mount the static files directory
app.mount("/audio", StaticFiles(directory="public/audio"), name="audio")

# Store job statuses in memory (use Redis in production)
jobs: Dict[str, dict] = {}

class YoutubeURL(BaseModel):
    url: str

@app.post("/generate-podcast")
async def generate_podcast(request: YoutubeURL):
    try:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "initializing",
            "audio_url": None,
            "error": None,
            "created_at": asyncio.get_event_loop().time()
        }
        
        # Start processing in background
        asyncio.create_task(process_youtube_url(job_id, request.url, jobs))
        
        return {"status": job_id}
    except Exception as e:
        logger.error(f"Error generating podcast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "status": job["status"],
        "audioUrl": job["audio_url"],
        "error": job["error"]
    }

@app.get("/health")
async def health_check():
    return JSONResponse({"status": "healthy"})

if __name__ == "__main__":
    # Configure uvicorn to ignore output and public/audio directories
    config = uvicorn.Config(
        "api:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=["./"],
        reload_excludes=["./output/*", "./public/*"],
    )
    server = uvicorn.Server(config)
    server.run()