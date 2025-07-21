from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
import os, httpx, logging

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AskRequest(BaseModel):
    query: str
    models: List[str]

@app.post("/ask")
async def ask_models(data: AskRequest):
    query = data.query
    models = data.models
    logger.info(f"Received query: {query}, Models: {models}")

    results: Dict[str, str] = {}
    summary_lines = []

    for model in models:
        try:
            if model == "gpt":
                api_key = os.getenv("GPT_API_KEY")
                url = os.getenv("GPT_API_URL", "https://api.openai.com/v1/chat/completions")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": query}]
                }

            elif model == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": query}]}]
                }

            elif model == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
                url = "https://api.deepseek.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": query}]
                }

            elif model == "mistral":
                api_key = os.getenv("MISTRAL_API_KEY")
                url = "https://api.mistral.ai/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "mistral-tiny",
                    "messages": [{"role": "user", "content": query}]
                }

            else:
                results[model] = f"[{model.upper()}] Unsupported model"
                continue

            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(url, headers=headers, json=payload)

            if response.status_code != 200:
                raise ValueError(f"[{model.upper()}] Error: {response.status_code} {response.text}")

            data = response.json()

            if model in ["gpt", "deepseek", "mistral"]:
                message = data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
            elif model == "gemini":
                message = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No content")
            else:
                message = "Unsupported format"

            results[model] = message

        except Exception as e:
            results[model] = f"[{model.upper()}] Error: {str(e)}"

    for m in models:
        summary_lines.append(f"{m.upper()}:\n{results.get(m, 'No response')}\n")

    return {
        "responses": results,
        "summary": "\n".join(summary_lines)
    }
