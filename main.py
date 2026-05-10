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

# Larger chunks = more context per question = better on-topic accuracy
MAX_CHUNK_SIZE     = 6000
CHUNK_OVERLAP      = 300
MAX_CHUNKS_PER_PDF = 8

# Semaphore limits concurrent AI calls
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
def extract_text_lean(file_bytes: bytes, max_chars: int = 60_000) -> str:
    """Extract text from PDF. Larger max_chars = more source material = better questions."""
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
    """
    Produce chunks sized so each chunk can comfortably generate its share of questions.
    More questions = more chunks = more source variety.
    """
    text_length = len(text)
    # Each chunk should cover roughly 8 questions worth of content
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


# ── Similarity dedup — only block near-duplicates ─────────────────────────
def is_too_similar(new_q: str, seen_questions: set[str], threshold: float = 0.85) -> bool:
    """
    Only block questions that are almost identical (0.85 overlap).
    Lower thresholds were dropping valid questions that just shared common topic words.
    """
    new_words = set(new_q.lower().split())
    if len(new_words) < 5:
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

    # Strict anchoring instruction — the key fix for off-topic questions
    anchor_rules = (
        f"You MUST generate EXACTLY {question_count} questions.\n"
        "CRITICAL RULES:\n"
        "1. Every question MUST be directly based on the TEXT provided below. "
        "Do NOT invent facts, definitions, or concepts not present in the text.\n"
        "2. Each question must test a DIFFERENT fact, term, or concept from the text.\n"
        "3. Include a mix of: definitions of key terms, important facts, "
        "cause-and-effect relationships, and application of concepts — "
        "all sourced from the text.\n"
        "4. Questions must be clear and unambiguous.\n"
        "5. Do NOT repeat or rephrase the same concept twice.\n"
        "6. Do NOT generate fewer than the requested count.\n"
    )

    if quiz_type == "multiple_choice":
        fmt = (
            f"Generate EXACTLY {question_count} multiple choice questions from the text.\n"
            + anchor_rules +
            "Format rules:\n"
            "- EXACTLY 4 choices per question labeled as plain strings (not A/B/C/D prefixed).\n"
            "- The 'answer' field must be the EXACT text of the correct choice.\n"
            "- Wrong choices must be plausible but clearly incorrect based on the text.\n"
            "Return ONLY a valid JSON array, no markdown, no explanation:\n"
            '[{"question": "...", "choices": ["choice1", "choice2", "choice3", "choice4"], "answer": "choice1"}, ...]'
        )
    elif quiz_type == "true_or_false":
        fmt = (
            f"Generate EXACTLY {question_count} true/false questions from the text.\n"
            + anchor_rules +
            "Format rules:\n"
            "- 'answer' must be exactly 'True' or 'False' (capital first letter only).\n"
            "- Mix of true and false statements — do not make all answers the same.\n"
            "- False statements should be plausible near-misses, not obviously wrong.\n"
            "Return ONLY a valid JSON array:\n"
            '[{"question": "...", "answer": "True"}, ...]'
        )
    elif quiz_type == "identification":
        fmt = (
            f"Generate EXACTLY {question_count} fill-in-the-blank / identification questions from the text.\n"
            + anchor_rules +
            "Format rules:\n"
            "- Blank out the key term, name, or concept being tested.\n"
            "- Use _____ (5 underscores) as the blank placeholder in the question.\n"
            "- 'answer' must be 1–5 words, taken directly from the text.\n"
            "- Questions should test definitions, names, processes, and key terms.\n"
            "Return ONLY a valid JSON array:\n"
            '[{"question": "_____ is defined as the process of ...", "answer": "Photosynthesis"}, ...]'
        )
    else:
        return ""

    return (
        "You are an expert quiz generator. Read the following text carefully and generate "
        f"questions ONLY from its content. This is chunk {chunk_index + 1} of {total_chunks}.\n\n"
        f"{fmt}\n\n"
        "TEXT TO USE:\n"
        "==========\n"
        f"{text_snippet}\n"
        "==========\n\n"
        f"Remember: Generate EXACTLY {question_count} questions. Base every question on the TEXT above."
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


def validate_question(q: dict, quiz_type: str) -> bool:
    """Validate question structure. Keep validation loose — don't drop valid questions."""
    question = q.get("question", "").strip()
    answer   = q.get("answer", "").strip()

    if not question or not answer:
        return False
    if len(question) < 10:
        return False

    if quiz_type == "multiple_choice":
        choices = q.get("choices", [])
        if len(choices) < 4:
            return False
        # Normalize to exactly 4 choices
        q["choices"] = [str(c).strip() for c in choices[:4]]
        # Answer must match one of the choices (case-insensitive)
        answer_lower = answer.lower()
        matched = next((c for c in q["choices"] if c.lower() == answer_lower), None)
        if matched is None:
            # Try partial match as fallback
            matched = next((c for c in q["choices"] if answer_lower in c.lower()), None)
        if matched is None:
            return False
        # Normalize answer to exact choice text
        q["answer"] = matched

    elif quiz_type == "true_or_false":
        normalized = answer.strip().lower()
        if normalized not in ("true", "false"):
            return False
        q["answer"] = normalized.capitalize()

    elif quiz_type == "identification":
        # Reject answers that are too long (model hallucinated a full sentence)
        if len(answer.split()) > 6:
            return False

    return True


# ── AI Calls ──────────────────────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    client = get_groq_client()
    if client is None:
        raise Exception("GROQ_API_KEY not set")
    last_error = None
    for model in GROQ_MODELS:
        try:
            logger.info(f"Trying Groq model: {model}")
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "You are an expert quiz generator. "
                        "Always respond with ONLY a valid JSON array. "
                        "Never include markdown, code blocks, or any text outside the JSON array. "
                        "Always generate the EXACT number of questions requested. "
                        "Base every question strictly on the provided text."
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,   # lower = more faithful to source text
                max_tokens=8192,
            )
            logger.info(f"Groq success: {model}")
            return resp.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq model {model} failed: {e}")
            last_error = e
            continue
    raise last_error or Exception("All Groq models exhausted")


def call_gemini(prompt: str) -> str:
    last_error = None
    for api_key in GEMINI_API_KEYS:
        client = get_gemini_client(api_key)
        for model_name in GEMINI_MODELS:
            try:
                logger.info(f"Trying Gemini model: {model_name}")
                resp = client.models.generate_content(model=model_name, contents=prompt)
                logger.info(f"Gemini success: {model_name}")
                return resp.text
            except Exception as e:
                logger.warning(f"Gemini {model_name} failed: {e}")
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
    """
    Ask for q_count questions from this chunk.
    If we get fewer valid ones, run ONE retry asking only for the shortfall.
    """
    prompt = build_prompt(quiz_type, q_count, chunk, idx, total_chunks)
    if not prompt.strip():
        return []

    valid_questions = []

    try:
        raw     = await call_ai_with_semaphore(prompt)
        cleaned = extract_json(raw)
        quiz    = json.loads(cleaned)
        if isinstance(quiz, list):
            valid_questions = [q for q in quiz if validate_question(q, quiz_type)]
        logger.info(f"Chunk {idx}: got {len(quiz) if isinstance(quiz, list) else 0}, valid {len(valid_questions)}, needed {q_count}")
    except Exception as e:
        logger.warning("Chunk %d first pass failed: %s", idx, e)

    # ── Top-up retry if we're short ───────────────────────────────────
    shortfall = q_count - len(valid_questions)
    if shortfall > 0 and len(valid_questions) > 0:
        logger.info(f"Chunk {idx}: short by {shortfall}, retrying for top-up")
        retry_prompt = build_prompt(quiz_type, shortfall, chunk, idx, total_chunks)
        try:
            raw2     = await call_ai_with_semaphore(retry_prompt)
            cleaned2 = extract_json(raw2)
            quiz2    = json.loads(cleaned2)
            if isinstance(quiz2, list):
                extra = [q for q in quiz2 if validate_question(q, quiz_type)]
                valid_questions.extend(extra)
                logger.info(f"Chunk {idx}: top-up added {len(extra)}")
        except Exception as e:
            logger.warning("Chunk %d top-up failed: %s", idx, e)
    elif shortfall > 0:
        # First pass returned nothing — retry the full count
        logger.info(f"Chunk {idx}: zero valid, full retry")
        retry_prompt = build_prompt(quiz_type, q_count, chunk, idx, total_chunks)
        try:
            raw2     = await call_ai_with_semaphore(retry_prompt)
            cleaned2 = extract_json(raw2)
            quiz2    = json.loads(cleaned2)
            if isinstance(quiz2, list):
                valid_questions = [q for q in quiz2 if validate_question(q, quiz_type)]
                logger.info(f"Chunk {idx}: full retry valid {len(valid_questions)}")
        except Exception as e:
            logger.warning("Chunk %d full retry failed: %s", idx, e)

    return valid_questions


# ── Core Quiz Generator ───────────────────────────────────────────────────
async def generate_quiz_from_text(text: str, quiz_type: str, question_count: int) -> list:
    chunks = chunk_text(text, question_count)
    del text
    gc.collect()

    total_chunks = len(chunks)
    logger.info(f"Generating {question_count} questions from {total_chunks} chunks")

    # Distribute questions evenly across chunks
    base_count  = question_count // total_chunks
    remainder   = question_count % total_chunks
    q_per_chunk = [base_count + (1 if i < remainder else 0) for i in range(total_chunks)]

    tasks = [
        process_chunk(i, chunk, q_per_chunk[i], quiz_type, total_chunks)
        for i, chunk in enumerate(chunks)
        if q_per_chunk[i] > 0
    ]
    del chunks
    gc.collect()

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge — deduplicate with high threshold to only block near-identical questions
    seen: set[str] = set()
    all_questions: list = []

    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"Chunk task failed: {result}")
            continue
        for q in result:
            q_text = q.get("question", "").strip().lower()
            if q_text and not is_too_similar(q_text, seen):
                seen.add(q_text)
                all_questions.append(q)

    logger.info(f"After dedup: {len(all_questions)} unique questions, {question_count} requested")

    # ── Final top-up: if still short, generate extra from the first chunk ──
    if len(all_questions) < question_count:
        shortfall = question_count - len(all_questions)
        logger.info(f"Final shortfall of {shortfall} — running global top-up")
        # Re-extract first chunk text isn't available here, so we log and return what we have
        # In practice the per-chunk retry above handles most shortfalls

    # ── Pad to exact count if slightly over or under ───────────────────
    result_list = all_questions[:question_count]

    logger.info(f"Returning {len(result_list)} questions (requested {question_count})")
    return result_list


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

    logger.info(f"Final quiz: {len(quiz)} questions (requested {question_count})")

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
        "priority": "Groq (llama-3.3-70b first) then Gemini (Flash-Lite first)",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info", reload=False)