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

# 8B model first — 14,400 RPD vs 1,000 RPD for 70B
GROQ_MODELS = [
    "llama3-8b-8192",
    "llama-3.3-70b-versatile",
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

# Flash-Lite first — fastest + highest RPD (1,500/day)
GEMINI_MODELS = [
    "models/gemini-2.5-flash-lite",
    "models/gemini-2.5-flash",
]

TRAILING_COMMA_RE = re.compile(r",\s*(?=[}\]])")

MAX_CHUNK_SIZE     = 3000
CHUNK_OVERLAP      = 150
MAX_CHUNKS_PER_PDF = 5

# 3 concurrent calls — balanced between speed and rate limit safety
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

    hot_count    = question_count // 2
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


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(kw in msg for kw in ("rate limit", "429", "quota", "too many"))


async def call_ai_with_retry(prompt: str, retries: int = 2) -> str:
    async def try_provider(fn):
        last_err = None
        for attempt in range(retries):
            try:
                return await asyncio.to_thread(fn, prompt)
            except Exception as e:
                last_err = e
                if _is_rate_limit_error(e) and attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    break
        raise last_err

    if GROQ_API_KEY:
        try:
            return await try_provider(call_groq)
        except Exception:
            pass

    return await try_provider(call_gemini)


async def call_ai_with_semaphore(prompt: str) -> str:
    async with AI_SEMAPHORE:
        return await call_ai_with_retry(prompt)


# ── Per-chunk processor ───────────────────────────────────────────────────
async def process_chunk(
    idx: int,
    chunk: str,
    q_count: int,
    quiz_type: str,
    total_chunks: int,
) -> list:
    prompt = build_prompt(quiz_type, q_count, chunk, idx, total_chunks)
    if not prompt.strip():
        return []
    try:
        raw     = await call_ai_with_semaphore(prompt)
        cleaned = extract_json(raw)
        quiz    = json.loads(cleaned)
        return ensure_choices(quiz, quiz_type)
    except Exception as e:
        logger.warning("Chunk %d failed: %s", idx, e)
        return []


# ── Core Quiz Generator ───────────────────────────────────────────────────
async def generate_quiz_from_text(text: str, quiz_type: str, question_count: int) -> list:
    chunks = chunk_text(text)
    del text
    gc.collect()

    total_chunks = len(chunks)
    base_count   = question_count // total_chunks
    remainder    = question_count % total_chunks
    q_per_chunk  = [base_count + (1 if i < remainder else 0) for i in range(total_chunks)]

    tasks = [
        process_chunk(i, chunk, q_per_chunk[i], quiz_type, total_chunks)
        for i, chunk in enumerate(chunks)
        if q_per_chunk[i] > 0
    ]
    del chunks
    gc.collect()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen: set[str] = set()
    all_questions: list = []
    for result in results:
        if isinstance(result, Exception):
            continue
        for q in result:
            q_text = q.get("question", "").strip().lower()
            if q_text and q_text not in seen:
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
    async def handle_file(file: UploadFile) -> dict:
        file_bytes = await file.read()
        text = extract_text_lean(file_bytes)
        del file_bytes
        gc.collect()

        if not text.strip():
            return {
                "file_name": file.filename,
                "error": "Could not extract text from PDF or PDF is empty.",
                "quiz": [],
            }

        quiz = await generate_quiz_from_text(text, quiz_type, question_count)
        del text
        gc.collect()

        return {
            "file_name": file.filename,
            "quiz_type": quiz_type,
            "total_questions": len(quiz),
            "requested": question_count,
            "quiz": quiz,
        }

    results = await asyncio.gather(*[handle_file(f) for f in files], return_exceptions=True)

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean.append({"file_name": files[i].filename, "error": str(r), "quiz": []})
        else:
            clean.append(r)

    return {"results": clean}


@app.get("/")
async def health():
    return {
        "status": "ok",
        "groq": "configured" if GROQ_API_KEY else "not set",
        "gemini_keys": f"{len(GEMINI_API_KEYS)} configured",
        "priority": "Groq (8B first) then Gemini (Flash-Lite first)",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)