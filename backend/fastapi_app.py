# =====================================================
# KrishiGPT Backend – FINAL STABLE (SUBMISSION READY)
# Chat → OpenAI
# Disease → YOLO + OpenAI
# Weather + Mandi + Voice intact
# =====================================================

import os
import json
import torch
import faiss
import numpy as np
import requests
import openai

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
from dotenv import load_dotenv

from ultralytics import YOLO
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------
# INIT
# -----------------------------------------------------
load_dotenv()

app = FastAPI(title="KrishiGPT Backend – Stable")

ROOT = os.path.dirname(os.path.dirname(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("▶ Using device:", DEVICE)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

# =====================================================
# RAG (LOADED BUT NOT USED FOR CHAT – SAFE)
# =====================================================
RAG_DIR = os.path.join(ROOT, "models", "rag")
FAISS_PATH = os.path.join(RAG_DIR, "faiss_index.idx")
CHUNKS_PATH = os.path.join(RAG_DIR, "chunks.jsonl")

index = faiss.read_index(FAISS_PATH)
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = [json.loads(x) for x in f]

embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

# =====================================================
# FINETUNED MODEL (LOADED – NOT USED FOR CHAT)
# =====================================================
BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"
ADAPTER_PATH = os.path.join(ROOT, "models", "llm", "adapter")

try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE)
    llm = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    llm.eval()
    print("✅ Finetuned model loaded (not used for chat)")
except:
    llm = None

# =====================================================
# YOLO
# =====================================================
YOLO_PATH = os.path.join(ROOT, "models", "yolo", "krishigpt_disease_cls.pt")
yolo = YOLO(YOLO_PATH)
print("✅ YOLO loaded")

# =====================================================
# SCHEMAS
# =====================================================
class QueryReq(BaseModel):
    query: str
    top_k: int = 3

class MandiReq(BaseModel):
    crop: str
    state: str

class WeatherReq(BaseModel):
    location: str

class TTSReq(BaseModel):
    text: str
    language: str = "en"

# =====================================================
# ✅ CHAT – OPENAI (FIXED)
# =====================================================
@app.post("/generate")
def generate(req: QueryReq):

    q = req.query.strip()
    if not q:
        return {"answer": "कृपया सवाल लिखें।"}

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are KrishiGPT, an Indian agriculture expert. "
                    "Answer clearly, practically, step-by-step in simple Hindi or Hinglish."
                )
            },
            {"role": "user", "content": q},
        ],
        temperature=0.3,
        max_tokens=300
    )

    return {"answer": resp.choices[0].message.content.strip()}

# =====================================================
# YOLO + OPENAI DISEASE ADVICE
# =====================================================
@app.post("/predict_disease")
async def predict_disease(file: UploadFile = File(...)):

    img_path = "tmp_leaf.jpg"
    with open(img_path, "wb") as f:
        f.write(await file.read())

    r = yolo(img_path)[0]
    cls_id = int(r.probs.top1)
    conf = round(float(r.probs.top1conf), 3)
    label = yolo.names[cls_id]

    advice = "⚠️ रोग पहचाना गया है। नजदीकी कृषि अधिकारी से सलाह लें।"

    try:
        ai = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Give crop disease treatment in simple Hindi."},
                {"role": "user", "content": f"{label} disease treatment, symptoms and prevention"}
            ],
            max_tokens=250,
            temperature=0.3
        )
        advice = ai.choices[0].message.content.strip()
    except:
        pass

    return {
        "disease": label,
        "confidence": conf,
        "description": advice
    }

# =====================================================
# SPEECH → TEXT
# =====================================================
@app.post("/speech_to_text")
async def speech_to_text(file: UploadFile = File(...)):

    audio = await file.read()
    with open("input.wav", "wb") as f:
        f.write(audio)

    res = openai.audio.transcriptions.create(
        model="whisper-1",
        file=open("input.wav", "rb")
    )

    return {"text": res.text, "language": "hi"}

# =====================================================
# TEXT → SPEECH
# =====================================================
@app.post("/text_to_speech")
def text_to_speech(req: TTSReq):

    speech = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=req.text
    )

    return StreamingResponse(BytesIO(speech.read()), media_type="audio/mpeg")

# =====================================================
# ✅ MANDI PRICES
# =====================================================
MANDI_API_KEY = os.getenv("MANDI_API_KEY")
MANDI_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

@app.post("/mandi_prices")
def mandi_prices(req: MandiReq):

    params = {
        "api-key": MANDI_API_KEY,
        "format": "json",
        "limit": 1,
        "filters[commodity]": req.crop,
        "filters[state]": req.state
    }

    data = requests.get(MANDI_URL, params=params).json()
    recs = data.get("records", [])

    if not recs:
        return {"text": "मंडी डेटा उपलब्ध नहीं है।"}

    r = recs[0]
    return {
        "text": (
            f"{req.crop} ({req.state})\n"
            f"Market: {r['market']}\n"
            f"Min: ₹{r['min_price']}\n"
            f"Max: ₹{r['max_price']}\n"
            f"Modal: ₹{r['modal_price']}"
        )
    }

# =====================================================
# ✅ WEATHER
# =====================================================
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY")

@app.post("/weather")
def weather(req: WeatherReq):

    if not OPENWEATHER_KEY:
        return {"text": f"{req.location}: 32°C, Sunny ☀ (demo)"}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": req.location,
        "appid": OPENWEATHER_KEY,
        "units": "metric"
    }

    data = requests.get(url, params=params).json()
    return {
        "text": f"{req.location}: {data['main']['temp']}°C, {data['weather'][0]['description']}"
    }

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"status": "KrishiGPT backend running ✅"}
