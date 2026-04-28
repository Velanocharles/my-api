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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq ──────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

# ── Gemini ────────────────────────────────────────────────────────────────
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

# ── Trailing comma regex (compiled once at module level) ──────────────────
# Matches a comma followed by optional whitespace then ] or }
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
            '[\n'
            '  {"question": "Why does ice float?", "choices": ["It is heavy", "Density decreases", "Melting point", "Freezes quickly"], "answer": "Density decreases"}\n'
            "]"
        )
    elif quiz_type == "true_or_false":
        format_instructions = (
            "Each question must include 'question' and 'answer' (True/False).\n"
            "Example:\n"
            '[{"question": "Water boils at 100°C.", "answer": "True"}]'
        )
    elif quiz_type == "identification":
        format_instructions = (
            "Each question must include 'question' and exact 'answer'.\n"
            "Example:\n"
            '[{"question": "Process plants use to convert sunlight to food?", "answer": "Photosynthesis"}]'
        )
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
            # 1. Try as-is
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
            # 2. Fix trailing commas — common Groq/Llama output issue
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

# ── Groq API Call ─────────────────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    if not GROQ_API_KEY:
        raise Exception("GROQ_API_KEY not set")

    client = Groq(api_key=GROQ_API_KEY)
    last_error = None

    for model in GROQ_MODELS:
        try:
            print(f"Trying Groq model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a quiz generator. "
                            "Always respond with ONLY a valid JSON array. "
                            "No markdown, no explanation, no extra text. "
                            "Do NOT use trailing commas in JSON."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            result = response.choices[0].message.content
            print(f"Groq success with model: {model}")
            return result
        except Exception as e:
            err = str(e)
            if any(code in err for code in ["429", "503", "rate_limit", "overloaded"]):
                print(f"Groq model {model} rate limited: {e}, trying next...")
                last_error = e
                time.sleep(1)
                continue
            elif any(code in err for code in ["model_not_active", "model_decommissioned", "404"]):
                print(f"Groq model {model} not available, trying next...")
                last_error = e
                continue
            else:
                raise

    raise last_error or Exception("All Groq models exhausted")

# ── Gemini API Call ───────────────────────────────────────────────────────
def call_gemini(prompt: str) -> str:
    last_error = None
    for api_key in GEMINI_API_KEYS:
        if not api_key:
            continue
        client = genai.Client(api_key=api_key)
        for model_name in GEMINI_MODELS:
            try:
                print(f"Trying Gemini key ...{api_key[-6:]} with model: {model_name}")
                response = client.models.generate_content(model=model_name, contents=prompt)
                print(f"Gemini success with model: {model_name}")
                return response.text
            except Exception as e:
                if any(code in str(e) for code in ["503", "429", "RESOURCE_EXHAUSTED"]):
                    print(f"Gemini model {model_name} failed: {e}, trying next...")
                    last_error = e
                    continue
                else:
                    raise
    raise last_error or Exception("All Gemini API keys and models exhausted")

# ── Primary: Groq first, Gemini fallback ─────────────────────────────────
def call_ai_with_retry(prompt: str, retries: int = 3, delay: int = 2) -> str:
    # 1. Try Groq first
    if GROQ_API_KEY:
        current_delay = delay
        for attempt in range(retries):
            try:
                return call_groq(prompt)
            except Exception as e:
                err = str(e)
                if any(code in err for code in ["429", "503", "rate_limit", "overloaded"]):
                    print(f"Groq attempt {attempt+1}/{retries} failed, retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    print(f"Groq non-retryable error: {e}, falling back to Gemini...")
                    break
        print("Groq exhausted, falling back to Gemini...")
    else:
        print("GROQ_API_KEY not set, using Gemini...")

    # 2. Fallback to Gemini
    current_delay = delay
    last_error = None
    for attempt in range(retries):
        try:
            return call_gemini(prompt)
        except Exception as e:
            if any(code in str(e) for code in ["503", "429", "RESOURCE_EXHAUSTED"]):
                last_error = e
                print(f"Gemini attempt {attempt+1}/{retries} failed, retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= 2
            else:
                raise

    raise last_error or Exception("Both Groq and Gemini exhausted all retries")

# ── Queue System ──────────────────────────────────────────────────────────
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

# ── FastAPI Endpoint ──────────────────────────────────────────────────────
@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...),
):
    try:
        file_bytes = await file.read()
        text = extract_text(file_bytes)
        if not text.strip():
            return {"error": "Could not extract text from PDF or PDF is empty."}

        prompt = build_prompt(quiz_type, question_count, text)
        if not prompt.strip():
            return {"error": "Failed to build prompt. PDF may be unreadable or empty."}

        position = quiz_queue.qsize() + 1
        future = asyncio.get_event_loop().create_future()
        await quiz_queue.put((call_ai_with_retry, [prompt], future))

        raw = await future
        if not raw or not raw.strip():
            return {"error": "Quiz generation failed: empty response from AI."}

        cleaned = extract_json(raw)
        try:
            quiz = json.loads(cleaned)
            quiz = ensure_choices(quiz, quiz_type)
        except json.JSONDecodeError as e:
            print(f"JSON decode failed. Raw response:\n{raw}")
            return {"error": f"Failed to parse quiz JSON: {str(e)}"}

        print(f"Quiz ready, first item preview: {quiz[0] if quiz else 'Empty'}")
        return {"quiz": quiz, "quiz_type": quiz_type, "position": position}

    except Exception as e:
        if any(code in str(e) for code in ["RESOURCE_EXHAUSTED", "429"]):
            return {"error": "AI quota exceeded. Please try later or upgrade plan."}
        print(f"Unexpected error in generate_quiz: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

# ── Health check ──────────────────────────────────────────────────────────
@app.get("/")
async def health():
    groq_status = "configured" if GROQ_API_KEY else "not set"
    gemini_keys = sum(1 for k in GEMINI_API_KEYS if k)
    return {
        "status":      "ok",
        "groq":        groq_status,
        "gemini_keys": f"{gemini_keys} configured",
        "priority":    "Groq then Gemini"
    }

# ── Server start ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)