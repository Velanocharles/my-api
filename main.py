from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
import json
import re
import os
import time
import math
import asyncio
from google import genai
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ───────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

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

MAX_CHUNK_SIZE = 2000
CHUNK_OVERLAP = 300
MAX_CHUNKS_PER_PDF = 15

# ── Semaphore to limit concurrent AI calls ────────────────────────────────
AI_SEMAPHORE = asyncio.Semaphore(2)  # Only 2 AI calls at a time to save memory


# ── Helper Functions ──────────────────────────────────────────────────────
def extract_text(file_bytes: bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_chunks = []
    for page in doc:
        text_chunks.append(page.get_text())
        if len(text_chunks) > 10:
            partial_text = "".join(text_chunks)
            text_chunks = [partial_text]
    return "".join(text_chunks)


def chunk_text(text: str, max_chunk_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP, max_chunks=MAX_CHUNKS_PER_PDF):
    text_length = len(text)
    if text_length <= max_chunk_size:
        return [text]

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
            "Each question MUST have exactly 4 choices in a 'choices' array.\n"
            "The correct answer must appear in 'choices' and also in 'answer'.\n"
            "Return ONLY valid JSON array, no markdown or extra text."
        )
    elif quiz_type == "true_or_false":
        format_instructions = (
            f"Generate exactly {question_count} TRUE OR FALSE questions.\n"
            "Each statement must have 'answer' field as 'True' or 'False'.\n"
            "Return ONLY valid JSON array."
        )
    elif quiz_type == "identification":
        format_instructions = (
            f"Generate exactly {question_count} FILL IN THE BLANK questions.\n"
            "Each question must have a single 'answer'.\n"
            "Return ONLY valid JSON array."
        )
    else:
        return ""

    return (
        f"You are a teacher creating a HOTS quiz.\n"
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
                cleaned = TRAILING_COMMA_RE.sub("", candidate)
                try:
                    json.loads(cleaned)
                    return cleaned
                except:
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


# ── AI Calls ──────────────────────────────────────────────────────────────
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
            last_error = e
            continue
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
                last_error = e
                continue
    raise last_error or Exception("All Gemini API keys and models exhausted")


async def call_ai_with_semaphore(prompt: str) -> str:
    async with AI_SEMAPHORE:
        return await asyncio.to_thread(call_ai_with_retry, prompt)


def call_ai_with_retry(prompt: str, retries: int = 3, delay: int = 2) -> str:
    # Retry Groq first
    if GROQ_API_KEY:
        current_delay = delay
        for attempt in range(retries):
            try:
                return call_groq(prompt)
            except Exception as e:
                time.sleep(current_delay)
                current_delay *= 2
    current_delay = delay
    for attempt in range(retries):
        try:
            return call_gemini(prompt)
        except Exception as e:
            time.sleep(current_delay)
            current_delay *= 2
    raise Exception("Both Groq and Gemini exhausted all retries")


# ── Generate Quiz ─────────────────────────────────────────────────────────
async def generate_quiz_from_pdf(text: str, quiz_type: str, question_count: int) -> list:
    chunks = chunk_text(text)
    total_chunks = len(chunks)
    all_questions = []

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
        raw = await call_ai_with_semaphore(prompt)
        cleaned = extract_json(raw)
        try:
            quiz = json.loads(cleaned)
            quiz = ensure_choices(quiz, quiz_type)
            all_questions.extend(quiz)
        except:
            continue
        # Free memory
        del chunk, prompt, raw, cleaned, quiz

    # remove duplicates
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


# ── Endpoints ───────────────────────────────────────────────────────────
@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...)
):
    file_bytes = await file.read()
    text = extract_text(file_bytes)
    if not text.strip():
        return {"error": "Could not extract text from PDF or PDF is empty."}

    quiz = await generate_quiz_from_pdf(text, quiz_type, question_count)
    return {"quiz": quiz, "quiz_type": quiz_type, "total_questions": len(quiz)}


@app.post("/generate-quiz-multiple")
async def generate_quiz_multiple(
    files: list[UploadFile] = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...)
):
    results = []

    for file in files:
        file_bytes = await file.read()
        text = extract_text(file_bytes)
        if not text.strip():
            results.append({
                "file_name": file.filename,
                "error": "Could not extract text from PDF or PDF is empty.",
                "quiz": []
            })
            continue

        quiz = await generate_quiz_from_pdf(text, quiz_type, question_count)
        results.append({
            "file_name": file.filename,
            "quiz_type": quiz_type,
            "total_questions": len(quiz),
            "quiz": quiz
        })

        # Free memory
        del file_bytes, text, quiz

    return {"results": results}


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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)