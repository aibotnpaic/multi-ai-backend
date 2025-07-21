from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
import logging
import httpx

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define request model
class AskRequest(BaseModel):
    query: str
    models: List[str]

@app.post("/ask")
async def ask_models(data: AskRequest):
    query = data.query
    models = data.models
    logger.info(f"Received query: {query}, Models: {models}")

    responses: Dict[str, str] = {}

    # === GPT (OpenAI) ===
    if "gpt" in models:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            responses["gpt"] = "[GPT] Missing API key."
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.7
                }
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    res = r.json()
                    responses["gpt"] = res.get("choices", [{}])[0].get("message", {}).get("content", "[GPT] No response.")
            except Exception as e:
                responses["gpt"] = f"[GPT] Error: {e}"

    # === Gemini ===
    if "gemini" in models:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            responses["gemini"] = "[GEMINI] Missing API key."
        else:
            try:
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": query}]
                        }
                    ]
                }
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(url, headers=headers, json=payload)
                    res = r.json()
                    responses["gemini"] = res.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "[GEMINI] No response.")
            except Exception as e:
                responses["gemini"] = f"[GEMINI] Error: {e}"

    # === DeepSeek ===
    if "deepseek" in models:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            responses["deepseek"] = "[DEEPSEEK] Missing API key."
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.7
                }
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
                    res = r.json()
                    responses["deepseek"] = res.get("choices", [{}])[0].get("message", {}).get("content", "[DEEPSEEK] No response.")
            except Exception as e:
                responses["deepseek"] = f"[DEEPSEEK] Error: {e}"

    # === Mistral ===
    if "mistral" in models:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            responses["mistral"] = "[MISTRAL] Missing API key."
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.7
                }
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
                    res = r.json()
                    responses["mistral"] = res.get("choices", [{}])[0].get("message", {}).get("content", "[MISTRAL] No response.")
            except Exception as e:
                responses["mistral"] = f"[MISTRAL] Error: {e}"

    # === Summary ===
    summary = "\n\n".join(f"{model.upper()}:\n{resp}" for model, resp in responses.items())
    return {
        "responses": responses,
        "summary": summary
    }
