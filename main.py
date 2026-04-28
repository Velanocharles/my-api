from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
import json
import re
import asyncio
import os
import time
from google import genai
from groq import Groq
import concurrent.futures

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq Configuration ─────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

# ── Gemini Configuration ───────────────────────────────────────────────────
GEMINI_API_KEYS = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
]
GEMINI_MODELS = [
    "models/gemini-2.5-pro",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
]

TRAILING_COMMA_RE = re.compile(r",\s*(?=[}\]])")

# ── Helper Functions ──────────────────────────────────────────────────────
def extract_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)

def build_prompt(quiz_type: str, question_count: int, text: str) -> str:
    snippet = text[:8000]
    if not snippet.strip():
        return ""

    base = (
        f"You are a teacher creating a HOTS (Higher Order Thinking Skills) quiz.\n"
        f"Return ONLY a valid JSON array with NO extra text, no markdown, no explanation.\n"
        f"Generate exactly {question_count} questions from the text below.\n\n"
    )

    if quiz_type == "multiple_choice":
        format_instructions = (
            "Each question must include exactly 4 choices in a 'choices' array.\n"
            "The correct answer must be in 'answer'.\n"
            "Example:\n"
            '[{"question": "Why does ice float?", "choices": ["It is heavy", "Density decreases", "Melting point", "Freezes quickly"], "answer": "Density decreases"}]'
        )
    elif quiz_type == "true_or_false":
        format_instructions = '[{"question": "Water boils at 100°C.", "answer": "True"}]'
    elif quiz_type == "identification":
        format_instructions = '[{"question": "Process plants use to convert sunlight to food?", "answer": "Photosynthesis"}]'
    else:
        return ""

    return f"{base}{format_instructions}\n\nText:\n{snippet}"

def extract_json(raw: str) -> str:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            candidate = raw[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
            cleaned = TRAILING_COMMA_RE.sub("", candidate)
            try:
                json.loads(cleaned)
                print("WARNING: Fixed trailing commas in JSON response")
                return cleaned
            except json.JSONDecodeError:
                continue
    return raw

def ensure_choices(quiz: list, quiz_type: str) -> list:
    if quiz_type != "multiple_choice":
        return quiz
    for q in quiz:
        if "choices" not in q or not q["choices"]:
            q["choices"] = ["Option 1", "Option 2", "Option 3", "Option 4"]
        if "answer" not in q or not q["answer"]:
            q["answer"] = q["choices"][0]
    return quiz

# ── Groq & Gemini API Calls ────────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not set")
    client = Groq(api_key=GROQ_API_KEY)
    last_error = None
    for model in GROQ_MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a quiz generator. Respond ONLY with valid JSON array."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["429", "503", "rate_limit", "overloaded"]):
                last_error = e
                time.sleep(1)
                continue
            elif any(code in err for code in ["model_not_active", "model_decommissioned", "404"]):
                last_error = e
                continue
            else:
                raise
    raise last_error or Exception("All Groq models exhausted")

def call_gemini(prompt: str) -> str:
    last_error = None
    for api_key in GEMINI_API_KEYS:
        if not api_key:
            continue
        client = genai.Client(api_key=api_key)
        for model_name in GEMINI_MODELS:
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                return response.text
            except Exception as e:
                if any(code in str(e) for code in ["503", "429", "RESOURCE_EXHAUSTED"]):
                    last_error = e
                    continue
                else:
                    raise
    raise last_error or Exception("All Gemini API keys and models exhausted")

def call_ai_with_retry(prompt: str, retries: int = 3, delay: int = 2) -> str:
    if GROQ_API_KEY:
        current_delay = delay
        for attempt in range(retries):
            try:
                return call_groq(prompt)
            except Exception as e:
                if any(code in str(e) for code in ["429", "503", "rate_limit", "overloaded"]):
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    break
    current_delay = delay
    last_error = None
    for attempt in range(retries):
        try:
            return call_gemini(prompt)
        except Exception as e:
            if any(code in str(e) for code in ["503", "429", "RESOURCE_EXHAUSTED"]):
                last_error = e
                time.sleep(current_delay)
                current_delay *= 2
            else:
                raise
    raise last_error or Exception("Both Groq and Gemini exhausted all retries")

# ── Multi-Worker Async Queue ──────────────────────────────────────────────
NUM_WORKERS = 4
quiz_queue = asyncio.Queue(maxsize=50)

async def process_quiz_worker(worker_id: int):
    loop = asyncio.get_running_loop()
    while True:
        func, args, future = await quiz_queue.get()
        try:
            result = await loop.run_in_executor(None, func, *args)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            quiz_queue.task_done()

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_running_loop()
    for i in range(NUM_WORKERS):
        loop.create_task(process_quiz_worker(i))
    print(f"Started {NUM_WORKERS} quiz workers")

# ── FastAPI Endpoint ──────────────────────────────────────────────────────
@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...),
):
    file_bytes = await file.read()
    text = extract_text(file_bytes)
    if not text.strip():
        return {"error": "Could not extract text from PDF or PDF is empty."}

    prompt = build_prompt(quiz_type, question_count, text)
    if not prompt.strip():
        return {"error": "Failed to build prompt. PDF may be unreadable or empty."}

    future = asyncio.get_running_loop().create_future()
    await quiz_queue.put((call_ai_with_retry, [prompt], future))
    raw = await future
    cleaned = extract_json(raw)
    try:
        quiz = json.loads(cleaned)
        quiz = ensure_choices(quiz, quiz_type)
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse quiz JSON: {str(e)}"}

    return {"quiz": quiz, "quiz_type": quiz_type, "position": quiz_queue.qsize() + 1}

# ── Health Check ─────────────────────────────────────────────────────────
@app.get("/")
async def health():
    groq_status = "configured" if GROQ_API_KEY else "not set"
    gemini_keys = sum(1 for k in GEMINI_API_KEYS if k)
    return {
        "status": "ok",
        "groq": groq_status,
        "gemini_keys": f"{gemini_keys} configured",
        "priority": "Groq then Gemini"
    }

# ── Run Server ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)