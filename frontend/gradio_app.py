import gradio as gr
import requests
import os
import io
from uuid import uuid4
from PIL import Image
from dotenv import load_dotenv

# -------------------------
# INIT
# -------------------------
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# -------------------------
# CHAT (DO NOT TOUCH ✅)
# -------------------------
def chat_fn(text):
    if not text or not text.strip():
        return "Please type something."

    try:
        r = requests.post(
            f"{API_URL}/generate",
            json={"query": text, "top_k": 5},
            timeout=180
        )
        r.raise_for_status()
        return r.json().get("answer", "")
    except Exception as e:
        return f"Backend error: {e}"

# -------------------------
# VOICE CHAT (BACKEND ONLY ✅)
# -------------------------
def voice_fn(audio_path):
    if audio_path is None:
        return "No audio received.", None

    # -------- STT (Backend) --------
    try:
        with open(audio_path, "rb") as f:
            stt = requests.post(
                f"{API_URL}/speech_to_text",
                files={"file": f},
                timeout=180
            )
        stt.raise_for_status()
        stt_data = stt.json()
        user_text = stt_data["text"]
        lang = stt_data.get("language", "en")
    except Exception as e:
        return f"STT error: {e}", None

    # -------- CHAT (RAG + Finetuned LLM) --------
    try:
        r = requests.post(
            f"{API_URL}/generate",
            json={"query": user_text, "top_k": 5},
            timeout=180
        )
        r.raise_for_status()
        answer = r.json().get("answer", "")
    except Exception as e:
        return f"Chat backend error: {e}", None

    # -------- TTS (Backend) --------
    try:
        tts = requests.post(
            f"{API_URL}/text_to_speech",
            json={"text": answer, "language": lang},
            timeout=180
        )
        tts.raise_for_status()

        out_audio = f"reply_{uuid4().hex}.mp3"
        with open(out_audio, "wb") as f:
            f.write(tts.content)
    except Exception as e:
        return answer + f"\n\n[TTS error: {e}]", None

    return answer, out_audio

# -------------------------
# DISEASE DETECTION (YOLO)
# -------------------------
def disease_fn(img: Image.Image):
    if img is None:
        return "Please upload a crop leaf image."

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    try:
        r = requests.post(
            f"{API_URL}/predict_disease",
            files={"file": ("leaf.jpg", buf, "image/jpeg")},
            timeout=120
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return f"Backend error: {e}"

    return (
        f"🌾 Disease Name: {data.get('disease')}\n"
        f"🎯 Confidence: {data.get('confidence')}\n\n"
        f"{data.get('description')}"
    )

# -------------------------
# MANDI PRICES
# -------------------------
def mandi_fn(crop, state):
    if not crop or not state:
        return "Please enter crop and state."

    try:
        r = requests.post(
            f"{API_URL}/mandi_prices",
            json={"crop": crop, "state": state},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("text", "")
    except Exception as e:
        return f"Backend error: {e}"

# -------------------------
# WEATHER
# -------------------------
def weather_fn(location):
    if not location or not location.strip():
        return "Please enter a location."

    try:
        r = requests.post(
            f"{API_URL}/weather",
            json={"location": location},
            timeout=30
        )
        r.raise_for_status()
        return r.json().get("text", "")
    except Exception as e:
        return f"Backend error: {e}"

# -------------------------
# UI
# -------------------------
with gr.Blocks(title="KrishiGPT – AI Assistant for Farmers") as app:

    gr.Markdown("""
    # 🌱 KrishiGPT  
    **AI Assistant for Farmers**  
    Chat • Voice • Disease Detection • Mandi Prices • Weather
    """)

    # ✅ CHAT TAB
    with gr.Tab("💬 Chat"):
        txt = gr.Textbox(label="Ask a question")
        out = gr.Textbox(
        label="Answer",
        lines=18,              # ⬆ big height
        max_lines=30,          # ⬆ scroll enable
        interactive=False,
)
        gr.Button("Ask").click(chat_fn, txt, out)

    # ✅ VOICE TAB
    with gr.Tab("🎙 Voice Assistant"):
        mic = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Speak your question"
        )
        voice_text = gr.Textbox(
            label="Text Answer",
            lines=6
        )
        voice_audio = gr.Audio(label="Voice Reply")
        gr.Button("Speak").click(
            voice_fn, mic, [voice_text, voice_audio]
        )

    # ✅ DISEASE DETECTION (WIDE OUTPUT ✅)
    with gr.Tab("🩺 Disease Detection"):
        img = gr.Image(type="pil", label="Upload crop leaf image")
        disease_out = gr.Textbox(
            label="Disease Prediction",
            lines=14,
            max_lines=25,
            interactive=False
        )
        gr.Button("Detect Disease").click(
            disease_fn, img, disease_out
        )

    # ✅ MANDI PRICES
    with gr.Tab("📊 Mandi Prices"):
        crop = gr.Textbox(label="Crop (e.g., Wheat)")
        state = gr.Textbox(label="State (e.g., Punjab)")
        mandi_out = gr.Textbox(label="Mandi Information", lines=10)
        gr.Button("Get Prices").click(
            mandi_fn, [crop, state], mandi_out
        )

    # ✅ WEATHER
    with gr.Tab("🌦 Weather"):
        loc = gr.Textbox(label="Location (City)")
        weather_out = gr.Textbox(label="Weather Info", lines=6)
        gr.Button("Check Weather").click(
            weather_fn, loc, weather_out
        )

app.launch()
