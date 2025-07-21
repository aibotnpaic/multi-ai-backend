from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
import os
import logging
import httpx

app = FastAPI()

# Enable logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model of expected request body
class AskRequest(BaseModel):
    query: str
    models: List[str]

@app.post("/ask")
async def ask_models(data: AskRequest):
    query = data.query
    models = data.models
    logger.info(f"Query: {query}, Models: {models}")

    responses: Dict[str, str] = {}
    
    if "gpt" in models:
        gpt_api_key = os.getenv("OPENAI_API_KEY")
        if not gpt_api_key:
            responses["gpt"] = "[GPT] Missing API key or URL."
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {gpt_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": query}]
                }
                async with httpx.AsyncClient() as client:
                    r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    res = r.json()
                    responses["gpt"] = res["choices"][0]["message"]["content"]
            except Exception as e:
                responses["gpt"] = f"[GPT] Error: {e}"

    if "gemini" in models:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            responses["gemini"] = "[GEMINI] Missing API key or URL."
        else:
            try:
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateContent?key={gemini_api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {"contents": [{"parts": [{"text": query}]}]}
                async with httpx.AsyncClient() as client:
                    r = await client.post(url, headers=headers, json=payload)
                    res = r.json()
                    responses["gemini"] = res["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                responses["gemini"] = f"[GEMINI] Error: {e}"

    if "deepseek" in models:
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if not deepseek_api_key:
            responses["deepseek"] = "[DEEPSEEK] Missing API key or URL."
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {deepseek_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": query}]
                }
                async with httpx.AsyncClient() as client:
                    r = await client.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
                    res = r.json()
                    responses["deepseek"] = res["choices"][0]["message"]["content"]
            except Exception as e:
                responses["deepseek"] = f"[DEEPSEEK] Error: {e}"

    if "mistral" in models:
        mistral_api_key = os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            responses["mistral"] = "[MISTRAL] Missing API key or URL."
        else:
            try:
                headers = {
                    "Authorization": f"Bearer {mistral_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": query}]
                }
                async with httpx.AsyncClient() as client:
                    r = await client.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload)
                    res = r.json()
                    responses["mistral"] = res["choices"][0]["message"]["content"]
            except Exception as e:
                responses["mistral"] = f"[MISTRAL] Error: {e}"

    summary = "\n\n".join(f"{model.upper()}:\n{content}" for model, content in responses.items())
    return {"responses": responses, "summary": summary}
