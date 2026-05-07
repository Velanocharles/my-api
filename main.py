from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
import json
import re
import os
import time
import math
import asyncio
import gc
from google import genai
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

GEMINI_API_KEYS = [k for k in [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
] if k]

GEMINI_MODELS = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-pro",
]

TRAILING_COMMA_RE = re.compile(r",\s*(?=[}\]])")

MAX_CHUNK_SIZE     = 1500
CHUNK_OVERLAP      = 100
MAX_CHUNKS_PER_PDF = 10

# ── Singleton AI clients ──────────────────────────────────────────────────
_groq_client: Groq | None = None
_gemini_clients: dict[str, genai.Client] = {}

def get_groq_client() -> Groq | None:
    global _groq_client
    if _groq_client is None and GROQ_API_KEY:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client

def get_gemini_client(api_key: str) -> genai.Client:
    if api_key not in _gemini_clients:
        _gemini_clients[api_key] = genai.Client(api_key=api_key)
    return _gemini_clients[api_key]

AI_SEMAPHORE = asyncio.Semaphore(1)


# ── Text Extraction ───────────────────────────────────────────────────────
def extract_text_lean(file_bytes: bytes, max_chars: int = 30_000) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    parts = []
    total = 0
    try:
        for page in doc:
            t = page.get_text()
            parts.append(t)
            total += len(t)
            page = None
            if total >= max_chars:
                break
    finally:
        doc.close()
    text = "".join(parts)
    del parts, file_bytes
    return text[:max_chars]


def chunk_text(text: str) -> list[str]:
    text_length = len(text)
    if text_length <= MAX_CHUNK_SIZE:
        return [text]

    chunk_size = MAX_CHUNK_SIZE
    if text_length / chunk_size > MAX_CHUNKS_PER_PDF:
        chunk_size = math.ceil(text_length / MAX_CHUNKS_PER_PDF)

    chunks = []
    start = 0
    while start < text_length and len(chunks) < MAX_CHUNKS_PER_PDF:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
    return chunks


# ── Prompt Builder ────────────────────────────────────────────────────────
def build_prompt(quiz_type: str, question_count: int, text_chunk: str,
                 chunk_index: int, total_chunks: int) -> str:
    text_snippet = text_chunk.strip()
    if not text_snippet:
        return ""

    # Split questions: 50% HOTS, 50% concept-based
    hot_count = question_count // 2
    normal_count = question_count - hot_count

    quality_rules = (
        f"STRICT RULES:\n"
        f"- Generate EXACTLY {question_count} questions.\n"
        f"- {hot_count} questions should be HOTS (Higher Order Thinking Skills) requiring analysis, evaluation, or application.\n"
        f"- {normal_count} questions should be concept-based, testing definitions or key ideas from the text.\n"
        "- Avoid trivial details like titles, dates, or formatting.\n"
        "- Do NOT repeat similar questions.\n"
        "- Each question should be clear, unambiguous, and based strictly on the provided text.\n"
        "- Vary question structures for engagement.\n"
    )

    if quiz_type == "multiple_choice":
        fmt = (
            f"Generate EXACTLY {question_count} multiple choice questions.\n"
            + quality_rules +
            "For normal questions:\n"
            "  - Focus on definitions, key terms, and core concepts.\n"
            "  - Provide EXACTLY 4 answer choices.\n"
            "  - The 3 incorrect choices should be plausible alternatives (conceptually related).\n"
            "For HOTS questions:\n"
            "  - Make questions that require reasoning, analysis, or application.\n"
            "Return ONLY a valid JSON array, no markdown or extra text:\n"
            '[{"question": "...", "choices": ["...", "...", "...", "..."], "answer": "..."}, ...]'
        )
    elif quiz_type == "true_or_false":
        fmt = (
            f"Generate EXACTLY {question_count} true/false questions.\n"
            + quality_rules +
            "For normal questions:\n"
            "  - Focus on statements about definitions or key concepts.\n"
            "  - Answer must be exactly 'True' or 'False'.\n"
            "For HOTS questions:\n"
            "  - Statements should require reasoning or evaluation.\n"
            "Return ONLY a valid JSON array:\n"
            '[{"question": "...", "answer": "True"}, ...]'
        )
    elif quiz_type == "identification":
        fmt = (
            f"Generate EXACTLY {question_count} fill-in-the-blank questions.\n"
            + quality_rules +
            "For normal questions:\n"
            "  - Blank key terms, definitions, or core concepts.\n"
            "For HOTS questions:\n"
            "  - Blanks that require reasoning or synthesis of ideas.\n"
            "Return ONLY a valid JSON array:\n"
            '[{"question": "_____ is defined as ...", "answer": "..."}, ...]'
        )
    else:
        return ""

    return (
        f"You are an expert educator creating high-quality quiz questions.\n"
        f"This is chunk {chunk_index + 1} of {total_chunks} from a study document.\n\n"
        f"{fmt}\n\n"
        f"TEXT TO USE:\n{text_snippet}"
    )


# ── JSON Extractor ────────────────────────────────────────────────────────
def extract_json(raw: str) -> str:
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    for sc, ec in [("[", "]"), ("{", "}")]:
        s = raw.find(sc)
        e = raw.rfind(ec)
        if s != -1 and e != -1 and e > s:
            candidate = raw[s:e + 1]
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
    valid = []
    for q in quiz:
        choices = q.get("choices", [])
        answer  = q.get("answer", "")
        if not choices or len(choices) < 4:
            continue
        # Drop questions where the AI returned placeholder choices
        if any(c.lower().startswith("option") for c in choices):
            continue
        if not answer or answer not in choices:
            continue
        valid.append(q)
    return valid


# ── AI Calls ──────────────────────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    client = get_groq_client()
    if client is None:
        raise Exception("GROQ_API_KEY not set")
    last_error = None
    for model in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert quiz generator. "
                        "You ALWAYS respond with ONLY a valid JSON array. "
                        "Never include markdown, explanations, or any text outside the JSON array."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=2048,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            continue
    raise last_error or Exception("All Groq models exhausted")


def call_gemini(prompt: str) -> str:
    last_error = None
    for api_key in GEMINI_API_KEYS:
        client = get_gemini_client(api_key)
        for model_name in GEMINI_MODELS:
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                return resp.text
            except Exception as e:
                last_error = e
                continue
    raise last_error or Exception("All Gemini keys and models exhausted")


def call_ai_with_retry(prompt: str, retries: int = 2, delay: int = 2) -> str:
    if GROQ_API_KEY:
        for attempt in range(retries):
            try:
                return call_groq(prompt)
            except Exception:
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
    for attempt in range(retries):
        try:
            return call_gemini(prompt)
        except Exception:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
    raise Exception("Both Groq and Gemini exhausted all retries")


async def call_ai_with_semaphore(prompt: str) -> str:
    async with AI_SEMAPHORE:
        return await asyncio.to_thread(call_ai_with_retry, prompt)


# ── Core Quiz Generator ───────────────────────────────────────────────────
async def generate_quiz_from_text(text: str, quiz_type: str, question_count: int) -> list:
    chunks = chunk_text(text)
    del text
    gc.collect()

    total_chunks   = len(chunks)
    base_count     = question_count // total_chunks
    remainder      = question_count % total_chunks
    q_per_chunk    = [base_count + (1 if i < remainder else 0) for i in range(total_chunks)]

    all_questions: list     = []
    seen_questions: set[str] = set()

    for idx, chunk in enumerate(chunks):
        q_count = q_per_chunk[idx]
        if q_count == 0:
            continue

        prompt = build_prompt(quiz_type, q_count, chunk, idx, total_chunks)
        if not prompt.strip():
            continue

        try:
            raw     = await call_ai_with_semaphore(prompt)
            cleaned = extract_json(raw)
            quiz    = json.loads(cleaned)
            quiz    = ensure_choices(quiz, quiz_type)

            for q in quiz:
                q_text = q.get("question", "").strip().lower()
                if q_text and q_text not in seen_questions:
                    seen_questions.add(q_text)
                    all_questions.append(q)

        except Exception:
            pass
        finally:
            del prompt, chunk
            gc.collect()

    del chunks

    result = all_questions[:question_count]
    del all_questions, seen_questions
    gc.collect()
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...),
):
    file_bytes = await file.read()
    text = extract_text_lean(file_bytes)
    del file_bytes
    gc.collect()

    if not text.strip():
        return {"error": "Could not extract text from PDF or PDF is empty."}

    quiz = await generate_quiz_from_text(text, quiz_type, question_count)
    del text
    gc.collect()

    return {
        "quiz": quiz,
        "quiz_type": quiz_type,
        "total_questions": len(quiz),
        "requested": question_count,
    }


@app.post("/generate-quiz-multiple")
async def generate_quiz_multiple(
    files: list[UploadFile] = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...),
):
    results = []

    for file in files:
        file_bytes = await file.read()
        text = extract_text_lean(file_bytes)
        del file_bytes
        gc.collect()

        if not text.strip():
            results.append({
                "file_name": file.filename,
                "error": "Could not extract text from PDF or PDF is empty.",
                "quiz": [],
            })
            del text
            continue

        quiz = await generate_quiz_from_text(text, quiz_type, question_count)
        del text
        gc.collect()

        results.append({
            "file_name": file.filename,
            "quiz_type": quiz_type,
            "total_questions": len(quiz),
            "requested": question_count,
            "quiz": quiz,
        })
        del quiz
        gc.collect()

    return {"results": results}


@app.get("/")
async def health():
    return {
        "status": "ok",
        "groq": "configured" if GROQ_API_KEY else "not set",
        "gemini_keys": f"{len(GEMINI_API_KEYS)} configured",
        "priority": "Groq then Gemini",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)