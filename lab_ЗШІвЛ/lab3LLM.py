import requests
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# =========================
# API KEY
# =========================

API_KEY = "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"

URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class ChatRequest(BaseModel):
    message: str


# =========================
# LLM function
# =========================

def generate_text(prompt):

    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(URL, headers=headers, json=payload)

    data = response.json()

    return data["choices"][0]["message"]["content"]


# =========================
# API endpoint
# =========================

@app.post("/chat")
def chat(req: ChatRequest):

    answer = generate_text(req.message)

    return {"answer": answer}


# =========================
# WEB PAGE
# =========================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):

    return templates.TemplateResponse(
        "chat.html",
        {"request": request}
    )