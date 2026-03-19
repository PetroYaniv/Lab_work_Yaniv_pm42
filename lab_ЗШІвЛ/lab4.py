import requests
import base64
import io

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")

class TextRequest(BaseModel):
    text: str


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/speak")
async def text_to_speech(data: TextRequest):

    url = "https://api.inworld.ai/tts/v1/voice"

    headers = {
        "Authorization": "Basic YWNLSHZQSFJmUnE5VUhkNWtjNDNSd0FxWkFaa3oweDA6Y2ZWUEhBVGNkZEZkblFQbTh6dEpDcm95MWk3SGw0WE1VQWJEcnlYWHRxS0FiV3R3ZnlGb0dPWjJ3N25hdkdzMQ==",
        "Content-Type": "application/json"
    }

    payload = {
        "text": data.text,
        "voiceId": "Clive",
        "modelId": "inworld-tts-1.5-max",
        "timestampType": "WORD",
        "speakingRate": 1,
        "temperature": 1
    }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    result = response.json()

    audio_bytes = base64.b64decode(result["audioContent"])

    return StreamingResponse(
        io.BytesIO(audio_bytes),
        media_type="audio/mpeg"
    )