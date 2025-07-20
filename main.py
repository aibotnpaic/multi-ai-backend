from fastapi import FastAPI, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import httpx, base64, os, json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Env Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

@app.post("/ask")
async def ask_models(
    query: str = Form(...),
    models: str = Form(...),
    image: Optional[UploadFile] = File(None)
):
    selected = json.loads(models)
    responses = {}
    async with httpx.AsyncClient() as client:
        # Optional Image Handling
        if image:
            image_bytes = await image.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            img_part = [{"inline_data": {"mime_type": image.content_type, "data": base64_image}},
                        {"text": query}]
        else:
            img_part = [{"text": query}]

        # GPT
        if "gpt" in selected:
            try:
                r = await client.post("https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                    json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": query}], "temperature": 0.7})
                responses["gpt"] = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                responses["gpt"] = f"Error: {str(e)}"

        # Gemini
        if "gemini" in selected:
            try:
                model = "gemini-pro-vision" if image else "gemini-pro"
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}",
                    json={"contents": [{"parts": img_part}]})
                responses["gemini"] = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                responses["gemini"] = f"Error: {str(e)}"

        # DeepSeek
        if "deepseek" in selected:
            try:
                r = await client.post("https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                    json={"model": "deepseek-chat", "messages": [{"role": "user", "content": query}]})
                responses["deepseek"] = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                responses["deepseek"] = f"Error: {str(e)}"

        # Mistral
        if "mistral" in selected:
            try:
                r = await client.post("https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "HTTP-Referer": "https://yourdomain.com",
                        "X-Title": "OmniAI"
                    },
                    json={"model": "mistralai/mistral-7b-instruct", "messages": [{"role": "user", "content": query}]})
                responses["mistral"] = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                responses["mistral"] = f"Error: {str(e)}"

        # Summary (GPT-based)
        try:
            summary_prompt = f"User query: {query}\n\n" + "\n".join([f"{k.upper()}: {v}" for k,v in responses.items()]) + "\n\nSummarize in 3-5 bullet points."
            r = await client.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": summary_prompt}], "temperature": 0.5})
            responses["summary"] = r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            responses["summary"] = f"Error: {str(e)}"

    return responses