from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
import json
import asyncio
import os
from google import genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEYS = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),  
]

def extract_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)

def build_prompt(quiz_type: str, question_count: int, text: str) -> str:
    snippet = text[:3000]  # limit to 3000 chars for memory efficiency
    if quiz_type == "multiple_choice":
        return f"""You are a teacher creating a HOTS quiz...
Text: {snippet}"""
    elif quiz_type == "true_or_false":
        return f"""You are a teacher creating a HOTS quiz...
Text: {snippet}"""
    elif quiz_type == "identification":
        return f"""You are a teacher creating a HOTS quiz...
Text: {snippet}"""

def call_gemini(prompt: str) -> str:
    models = [
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
    ]

    last_error = None
    for api_key in API_KEYS:
        if not api_key:
            continue
        client = genai.Client(api_key=api_key)
        for model_name in models:
            try:
                print(f"⏳ Trying key ...{api_key[-6:]} with model: {model_name}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                print(f"✅ Success with model: {model_name}")
                return response.text
            except Exception as e:
                if "429" in str(e) or "404" in str(e):
                    last_error = e
                    continue
                else:
                    raise
    raise last_error or Exception("All API keys and models exhausted!")

# ---------------- Queue System for Starter Plan ----------------
quiz_queue = asyncio.Queue()

async def process_quiz_queue():
    while True:
        func, args, future = await quiz_queue.get()
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, func, *args)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            quiz_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_quiz_queue())

@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...)
):
    file_bytes = await file.read()
    text = extract_text(file_bytes)
    if not text.strip():
        return {"error": "Could not extract text from PDF"}

    prompt = build_prompt(quiz_type, question_count, text)

    # Add to queue for sequential processing
    future = asyncio.get_event_loop().create_future()
    await quiz_queue.put((call_gemini, [prompt], future))
    raw = await future  # wait until the quiz is generated

    raw = raw.replace("```json", "").replace("```", "").strip()
    quiz = json.loads(raw)
    return {"quiz": quiz, "quiz_type": quiz_type}

# ---------------- Render Starter-friendly server start ----------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",  # replace 'main' with your filename if different
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )