# api_server.py
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

class QuestRequest(BaseModel):
    mode: int
    coords: Optional[List[float]]
    stt_prompt: str

@app.post("/quest-query")
async def handle_quest(image: UploadFile, request: QuestRequest):
    pass