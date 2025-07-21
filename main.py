from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import httpx
import os

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class AskRequest(BaseModel):
    query: str
    models: List[str]

# Get API keys from environment variables
API_KEYS = {
    "gpt": os.getenv("GPT_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY")
}

# Dummy API endpoints for each provider (replace with actual)
MODEL_ENDPOINTS = {
    "gpt": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions"
}

# Dummy request builders (adjust per real API)
def build_payload(model_name: str, query: str):
    if model_name in {"gpt", "deepseek", "mistral"}:
        return {
            "model": "gpt-3.5-turbo",  # Adjust model per provider
            "messages": [{"role": "user", "content": query}],
            "temperature": 0.7
        }
    elif model_name == "gemini":
        return {
            "contents": [{"parts": [{"text": query}]}]
        }
    else:
        return {}

def build_headers(model_name: str, api_key: str):
    if model_name == "gemini":
        return {"Content-Type": "application/json"}
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

# Call each model
async def call_model(model: str, query: str) -> str:
    api_key = API_KEYS.get(model)
    url = MODEL_ENDPOINTS.get(model)

    if not api_key or not url:
        return f"[{model.upper()}] Missing API key or URL."

    payload = build_payload(model, query)
    headers = build_headers(model, api_key)

    # Add API key to URL for Gemini
    if model == "gemini":
        url += f"?key={api_key}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(url, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()

            # Extract response text based on model
            if model == "gemini":
                return data['candidates'][0]['content']['parts'][0]['text']
            else:
                return data['choices'][0]['message']['content']
    except Exception as e:
        return f"[{model.upper()}] Error: {str(e)}"

# Endpoint
@app.post("/ask")
async def ask_models(request: AskRequest):
    logger.info(f"Received query: {request.query} for models: {request.models}")

    results = {}
    for model in request.models:
        answer = await call_model(model, request.query)
        results[model] = answer

    # Combine for summary
    summary = "\n\n".join(f"{m.upper()}:\n{a}" for m, a in results.items())

    return {
        "responses": results,
        "summary": summary
    }
