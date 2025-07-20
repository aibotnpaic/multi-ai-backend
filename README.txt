# Deploy to Render

1. Go to https://render.com
2. Click "New Web Service"
3. Connect your GitHub repo (or drag these files into a new GitHub repo)
4. Choose:
   - Environment: Python 3.11+
   - Start command: uvicorn main:app --host 0.0.0.0 --port 10000
5. Add these Environment Variables in the Render dashboard:
   - OPENAI_API_KEY
   - GEMINI_API_KEY
   - DEEPSEEK_API_KEY
   - MISTRAL_API_KEY
6. Click Deploy. Your API will be live at:

   https://your-service-name.onrender.com/ask