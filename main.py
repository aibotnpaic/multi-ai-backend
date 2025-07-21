from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import httpx
import logging
import os

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Request model
class AskRequest(BaseModel):
    query: str
    models: List[str]

# Your environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

@app.post("/ask")
async def ask_models(data: AskRequest):
    query = data.query
    models = data.models
    logger.info(f"Received query: {query}, Models: {models}")

    responses: Dict[str, str] = {}

    async with httpx.AsyncClient(timeout=30.0) as client:
        if "gpt" in models:
            if not OPENAI_API_KEY:
                responses["gpt"] = "[GPT] Missing API key or URL."
            else:
                try:
                    gpt_response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "gpt-3.5-turbo",
                            "messages": [{"role": "user", "content": query}],
                        },
                    )
                    gpt_data = gpt_response.json()
                    responses["gpt"] = gpt_data["choices"][0]["message"]["content"]
                except Exception as e:
                    responses["gpt"] = f"[GPT] Error: {str(e)}"

        if "gemini" in models:
            if not GEMINI_API_KEY:
                responses["gemini"] = "[GEMINI] Missing API key."
            else:
                try:
                    gemini_response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}",
                        json={"contents": [{"parts": [{"text": query}]}]},
                    )
                    gemini_data = gemini_response.json()
                    responses["gemini"] = gemini_data["candidates"][0]["content"]["parts"][0]["text"]
                except Exception as e:
                    responses["gemini"] = f"[GEMINI] Error: {str(e)}"

        if "deepseek" in models:
            if not DEEPSEEK_API_KEY:
                responses["deepseek"] = "[DEEPSEEK] Missing API key."
            else:
                try:
                    deepseek_response = await client.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": query}],
                        },
                    )
                    deepseek_data = deepseek_response.json()
                    responses["deepseek"] = deepseek_data["choices"][0]["message"]["content"]
                except Exception as e:
                    responses["deepseek"] = f"[DEEPSEEK] Error: {str(e)}"

        if "mistral" in models:
            if not MISTRAL_API_KEY:
                responses["mistral"] = "[MISTRAL] Missing API key."
            else:
                try:
                    mistral_response = await client.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {MISTRAL_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "mistral-tiny",
                            "messages": [{"role": "user", "content": query}],
                        },
                    )
                    mistral_data = mistral_response.json()
                    responses["mistral"] = mistral_data["choices"][0]["message"]["content"]
                except Exception as e:
                    responses["mistral"] = f"[MISTRAL] Error: {str(e)}"

    # Combine summary
    summary = "\n\n".join(
        f"{model.upper()}:\n{response}" for model, response in responses.items()
    )

    return {
        "responses": responses,
        "summary": summary,
    }
