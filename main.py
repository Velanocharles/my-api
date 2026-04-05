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
]

def extract_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)

def build_prompt(quiz_type: str, question_count: int, text: str) -> str:
    if quiz_type == "multiple_choice":
        return f"""You are a teacher creating a HOTS (Higher Order Thinking Skills) quiz.
Generate {question_count} multiple choice questions from this text.

STRICT RULES:
- Questions must require ANALYSIS, EVALUATION, or APPLICATION — not just recall
- Use question starters like: "Why", "How", "What would happen if", "Which best explains", "What is the most likely reason"
- Each choice must be SHORT (1-5 words only)
- All 4 choices must be plausible but only one is correct
- Answer must EXACTLY match one of the choices word for word
- No letters (A, B, C, D) in choices
- Return ONLY a valid JSON array, no markdown, no extra text

Example:
[{{"question": "Why does water expand when frozen?", "choices": ["Molecules slow down", "Hydrogen bonds form", "Density increases", "Heat is absorbed"], "answer": "Hydrogen bonds form"}}]

Text: {text[:3000]}"""

    elif quiz_type == "true_or_false":
        return f"""You are a teacher creating a HOTS (Higher Order Thinking Skills) quiz.
Generate {question_count} true or false questions from this text.

STRICT RULES:
- Questions must require ANALYSIS or EVALUATION — not just memorization
- Avoid simple fact-based questions
- Use statements that require the student to think critically
- Mix true and false answers roughly equally
- Answer must be exactly "True" or "False"
- Return ONLY a valid JSON array, no markdown, no extra text

Example:
[{{"question": "Increasing temperature always increases the rate of a chemical reaction.", "choices": ["True", "False"], "answer": "False"}}]

Text: {text[:3000]}"""

    elif quiz_type == "identification":
        return f"""You are a teacher creating a HOTS (Higher Order Thinking Skills) quiz.
Generate {question_count} identification questions from this text.

STRICT RULES:
- Questions must require the student to APPLY or ANALYZE concepts, not just recall terms
- Use question starters like: "What term describes", "Identify the concept", "What process explains"
- Answer must be a SHORT and SPECIFIC phrase (1-4 words only)
- Answer must come directly from the text
- Return ONLY a valid JSON array, no markdown, no extra text

Example:
[{{"question": "What process describes plants converting sunlight into food?", "answer": "Photosynthesis"}}]

Text: {text[:3000]}"""

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
                    print(f"❌ {model_name} failed, trying next...")
                    last_error = e
                    continue
                else:
                    raise

    raise last_error or Exception("All API keys and models exhausted!")

@app.post("/generate-quiz")
async def generate_quiz(
    file: UploadFile = File(...),
    quiz_type: str = Form(...),
    question_count: int = Form(...)
):
    try:
        file_bytes = await file.read()
        print(f"✅ File received: {file.filename}, {len(file_bytes)} bytes")

        text = extract_text(file_bytes)
        print(f"✅ Text extracted: {len(text)} characters")

        if not text.strip():
            return {"error": "Could not extract text from PDF"}

        prompt = build_prompt(quiz_type, question_count, text)
        print(f"✅ Calling Gemini for: {quiz_type}, {question_count} questions")

        raw = await asyncio.get_event_loop().run_in_executor(
            None, call_gemini, prompt
        )

        print(f"✅ Gemini response: {raw[:300]}")

        raw = raw.replace("```json", "").replace("```", "").strip()
        quiz = json.loads(raw)

        print(f"✅ Quiz generated: {len(quiz)} questions")
        return {"quiz": quiz, "quiz_type": quiz_type}

    except Exception as e:
        import traceback
        print(f"❌ ERROR: {str(e)}")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)