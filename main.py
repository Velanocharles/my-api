from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import fitz
import json
import re
import os
import asyncio
import gc
import math
import logging
from google import genai
from groq import Groq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
]

TRAILING_COMMA_RE = re.compile(r",\s*(?=[}\]])")

MAX_CHUNK_SIZE = 4000
CHUNK_OVERLAP = 200
MAX_CHUNKS_PER_PDF = 6
OVERGENERATE_FACTOR = 1.6
AI_SEMAPHORE = asyncio.Semaphore(3)

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

# ── Text Extraction ───────────────────────────────────────────────────────
def extract_text_lean(file_bytes: bytes, max_chars: int = 40000) -> str:
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

def chunk_text(text: str, question_count: int) -> list[str]:
    text_length = len(text)
    desired_chunks = max(1, min(MAX_CHUNKS_PER_PDF, math.ceil(question_count / 8)))
    if text_length <= MAX_CHUNK_SIZE or desired_chunks == 1:
        return [text]
    chunk_size = max(MAX_CHUNK_SIZE, math.ceil(text_length / desired_chunks))
    chunks = []
    start = 0
    while start < text_length and len(chunks) < desired_chunks:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
    return chunks if chunks else [text]

# ── Similarity Check ──────────────────────────────────────────────────────
def is_too_similar(new_q: str, seen_questions: set[str], threshold: float = 0.75) -> bool:
    new_words = set(new_q.lower().split())
    if len(new_words) < 4:
        return False
    for seen_q in seen_questions:
        seen_words = set(seen_q.lower().split())
        if not seen_words:
            continue
        overlap = len(new_words & seen_words) / max(len(new_words), len(seen_words))
        if overlap >= threshold:
            return True
    return False

# ── Prompt Builder ────────────────────────────────────────────────────────
def build_prompt(quiz_type: str, question_count: int, text_chunk: str,
                 chunk_index: int, total_chunks: int) -> str:
    text_snippet = text_chunk.strip()
    if not text_snippet:
        return ""
    hot_count = question_count // 2
    normal_count = question_count - hot_count
    quality_rules = (
        f"CRITICAL: Generate EXACTLY {question_count} questions.\n"
        f"- {hot_count} HOTS questions\n"
        f"- {normal_count} concept-based questions\n"
        "- Each question must be unique and clear.\n"
        "- Return ONLY a valid JSON array."
    )
    if quiz_type == "multiple_choice":
        fmt = (
            f"Generate EXACTLY {question_count} multiple choice questions.\n"
            + quality_rules +
            "\nEach question must have 4 choices with the correct answer included.\n"
            "Return JSON array only:\n"
            '[{"question": "...", "choices": ["A","B","C","D"], "answer": "A"}, ...]'
        )
    elif quiz_type == "true_or_false":
        fmt = (
            f"Generate EXACTLY {question_count} true/false questions.\n"
            + quality_rules +
            "\nAnswer must be 'True' or 'False'.\n"
            "Return JSON array only:\n"
            '[{"question": "...", "answer": "True"}, ...]'
        )
    elif quiz_type == "identification":
        fmt = (
            f"Generate EXACTLY {question_count} fill-in-the-blank questions.\n"
            + quality_rules +
            "\nAnswer must be 1-5 words.\n"
            "Return JSON array only:\n"
            '[{"question": "_____ is ...", "answer": "term"}, ...]'
        )
    else:
        return ""
    return f"You are an expert educator. Chunk {chunk_index+1}/{total_chunks}.\n\n{fmt}\n\nTEXT:\n{text_snippet}"

# ── JSON Extractor ───────────────────────────────────────────────────────
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

def validate_question(q: dict, quiz_type: str) -> bool:
    question = q.get("question", "").strip()
    answer = q.get("answer", "").strip()
    if not question or not answer or len(question) < 10:
        return False
    if quiz_type == "multiple_choice":
        choices = q.get("choices", [])
        if not choices or len(choices) != 4:
            return False
        if not any(c.strip().lower() == answer.lower() for c in choices):
            return False
    elif quiz_type == "true_or_false":
        if answer.lower() not in ("true", "false"):
            return False
        q["answer"] = answer.capitalize()
    return True

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
                    {"role": "system", "content": "You are an expert quiz generator. Return only JSON array."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            continue
    raise last_error

async def call_ai_with_semaphore(prompt: str) -> str:
    async with AI_SEMAPHORE:
        return await asyncio.to_thread(call_groq, prompt)

async def process_chunk(idx: int, chunk: str, q_count: int, quiz_type: str, total_chunks: int) -> list:
    overcount = math.ceil(q_count * OVERGENERATE_FACTOR)
    prompt = build_prompt(quiz_type, overcount, chunk, idx, total_chunks)
    if not prompt.strip():
        return []
    try:
        raw = await call_ai_with_semaphore(prompt)
        cleaned = extract_json(raw)
        quiz = json.loads(cleaned)
        if not isinstance(quiz, list):
            return []
        valid = [q for q in quiz if validate_question(q, quiz_type)]
        logger.info(f"Chunk {idx}: got {len(quiz)} total, {len(valid)} valid")
        return valid
    except Exception as e:
        logger.warning(f"Chunk {idx} failed: {e}")
        return []

async def generate_quiz_from_text(text: str, quiz_type: str, question_count: int) -> list:
    chunks = chunk_text(text, question_count)
    del text
    gc.collect()
    total_chunks = len(chunks)
    base_count = question_count // total_chunks
    remainder = question_count % total_chunks
    q_per_chunk = [base_count + (1 if i < remainder else 0) for i in range(total_chunks)]
    tasks = [process_chunk(i, chunk, q_per_chunk[i], quiz_type, total_chunks)
             for i, chunk in enumerate(chunks) if q_per_chunk[i] > 0]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    seen = set()
    all_questions = []
    for result in results:
        if isinstance(result, Exception):
            continue
        for q in result:
            q_text = q.get("question", "").strip().lower()
            if q_text and not is_too_similar(q_text, seen):
                seen.add(q_text)
                all_questions.append(q)
    # Top-up pass if needed
    if len(all_questions) < question_count and len(all_questions) > 0:
        shortfall = question_count - len(all_questions)
        top_chunk = chunks[0]
        top_questions = await process_chunk(0, top_chunk, shortfall, quiz_type, total_chunks)
        for q in top_questions:
            q_text = q.get("question", "").strip().lower()
            if q_text and not is_too_similar(q_text, seen):
                seen.add(q_text)
                all_questions.append(q)
    return all_questions[:question_count]

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
        return {"error": "PDF text extraction failed"}
    quiz = await generate_quiz_from_text(text, quiz_type, question_count)
    del text
    gc.collect()
    return {"quiz": quiz, "quiz_type": quiz_type, "total_questions": len(quiz), "requested": question_count}

@app.get("/")
async def health():
    return {
        "status": "ok",
        "groq": "configured" if GROQ_API_KEY else "not set",
        "gemini_keys": f"{len(GEMINI_API_KEYS)} configured",
        "priority": "Groq first, then Gemini"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)