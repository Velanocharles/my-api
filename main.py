from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
import json
import re
import asyncio
import os
import time
import math
from google import genai
from groq import Groq

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


def chunk_text(text: str, max_chunk_size: int = 3000, overlap: int = 500, max_chunks: int = 10) -> list[str]:
    """
    Split text into overlapping chunks for full coverage.
    Dynamically adjusts chunk size to limit number of chunks to max_chunks.
    """
    text_length = len(text)
    if text_length <= max_chunk_size:
        return [text]

    # Adjust chunk_size if text is too long
    chunk_size = max_chunk_size
    if text_length / chunk_size > max_chunks:
        chunk_size = math.ceil(text_length / max_chunks)

    chunks = []
    start = 0
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_prompt(quiz_type: str, question_count: int, text_chunk: str, chunk_index: int, total_chunks: int) -> str:
    text_snippet = text_chunk.strip()
    if not text_snippet:
        return ""

    if quiz_type == "multiple_choice":
        format_instructions = (
            f"Generate exactly {question_count} MULTIPLE CHOICE questions.\n"
            "Each question must focus on a different concept or fact from this chunk.\n"
            "Each question MUST have exactly 4 choices in a 'choices' array.\n"
            "The correct answer must appear in the 'choices' array and also in 'answer'.\n"
            "Return ONLY a valid JSON array. No markdown, no explanation, no extra text."
        )
    elif quiz_type == "true_or_false":
        format_instructions = (
            f"Generate exactly {question_count} TRUE OR FALSE questions.\n"
            "Each statement must focus on a different concept or fact from this chunk.\n"
            "The 'answer' field must be ONLY 'True' or 'False'.\n"
            "Do NOT include a 'choices' field.\n"
            "Return ONLY a valid JSON array. No markdown, no explanation, no extra text."
        )
    elif quiz_type == "identification":
        format_instructions = (
            f"Generate exactly {question_count} FILL IN THE BLANK questions.\n"
            "Each question must focus on a different concept or term from this chunk.\n"
            "Return ONLY a valid JSON array. No markdown, no explanation, no extra text."
        )
    else:
        return ""

    return (
        f"You are a teacher creating a HOTS (Higher Order Thinking Skills) quiz.\n"
        f"Chunk {chunk_index + 1} of {total_chunks}\n"
        f"{format_instructions}\n\n"
        f"Text to base questions on:\n{text_snippet}"
    )


def extract_json(raw: str) -> str:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    for start_char, end_char in [("[", "]"), ("{", "}")] :
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
MAX_QUEUE_SIZE = 50
quiz_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)


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


# ── Generate Quiz from PDF ────────────────────────────────────────────────
async def generate_quiz_from_pdf(text: str, quiz_type: str, question_count: int) -> list:
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    all_questions = []

    # Allocate questions per chunk
    base_count = question_count // total_chunks
    remainder = question_count % total_chunks
    questions_per_chunk = [base_count] * total_chunks
    for i in range(remainder):
        questions_per_chunk[i] += 1

    for idx, chunk in enumerate(chunks):
        q_count = questions_per_chunk[idx]
        if q_count == 0:
            continue
        prompt = build_prompt(quiz_type, q_count, chunk, idx, total_chunks)
        if not prompt.strip():
            continue
        future = asyncio.get_running_loop().create_future()
        await quiz_queue.put((call_ai_with_retry, [prompt], future))
        raw = await future
        cleaned = extract_json(raw)
        try:
            quiz = json.loads(cleaned)
            quiz = ensure_choices(quiz, quiz_type)
            all_questions.extend(quiz)
        except json.JSONDecodeError:
            continue

    # Remove duplicates and trim to requested question_count
    seen = set()
    unique_questions = []
    for q in all_questions:
        q_text = q.get("question", "")
        if q_text not in seen:
            seen.add(q_text)
            unique_questions.append(q)
        if len(unique_questions) >= question_count:
            break

    return unique_questions


# ── FastAPI Endpoint ─────────────────────────────────────────────────────
@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...)
):
    if quiz_queue.qsize() >= MAX_QUEUE_SIZE:
        return {"error": "Server is busy, please try again later."}

    file_bytes = await file.read()
    text = extract_text(file_bytes)
    if not text.strip():
        return {"error": "Could not extract text from PDF or PDF is empty."}

    quiz = await generate_quiz_from_pdf(text, quiz_type, question_count)
    return {"quiz": quiz, "quiz_type": quiz_type, "total_questions": len(quiz)}


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