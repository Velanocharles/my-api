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

API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

def extract_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)

def build_prompt(quiz_type: str, question_count: int, text: str) -> str:
    if quiz_type == "multiple_choice":
        return f"""Generate {question_count} multiple choice questions from this text.
Return ONLY a valid JSON array, no markdown, no extra text.
IMPORTANT:
- choices must be the FULL answer text, NOT letters like A, B, C, D
- answer must EXACTLY match one of the choices word for word
- do NOT include letters or numbering in the choices
Example format:
[{{"question": "What is the capital of France?", "choices": ["Paris", "London", "Berlin", "Madrid"], "answer": "Paris"}}]
Text: {text[:3000]}"""

    elif quiz_type == "true_or_false":
        return f"""Generate {question_count} true or false questions from this text.
Return ONLY a valid JSON array, no markdown, no extra text.
IMPORTANT: answer must be exactly "True" or "False"
Example format:
[{{"question": "The Earth is round.", "choices": ["True", "False"], "answer": "True"}}]
Text: {text[:3000]}"""

    elif quiz_type == "identification":
        return f"""Generate {question_count} identification questions from this text.
Return ONLY a valid JSON array, no markdown, no extra text.
IMPORTANT: answer must be a short exact phrase from the text
Example format:
[{{"question": "Who discovered gravity?", "answer": "Isaac Newton"}}]
Text: {text[:3000]}"""

def call_gemini(prompt: str) -> str:
    models = [
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-flash",
        "models/gemini-2.5-flash",
    ]

    last_error = None
    for model_name in models:
        try:
            print(f"⏳ Trying model: {model_name}")
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

    raise last_error

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