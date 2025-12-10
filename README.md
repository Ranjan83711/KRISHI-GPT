Got it. Tum bilkul sahi bol rahe ho — **README me clearly “how to run” steps + pipeline flow missing tha**.
Below is an **UPDATED & COMPLETE README** section set that you can **directly append or replace** in your README.
No code blocks, no OpenAI chat mention, and **simple examiner-friendly language**.

---

## 🚀 How to Run the Project (Step-by-Step)

### ✅ 1. System Requirements

* Python 3.9+
* NVIDIA GPU (4GB sufficient)
* CUDA enabled environment
* Windows / Linux
* Internet connection (for APIs)

---

### ✅ 2. Environment Setup

1. Create and activate a Python virtual environment
2. Install all required dependencies using the provided requirements file
3. Place trained models in correct folders:

   * Fine-tuned LLM adapters → models/llm/adapter
   * FAISS index & chunks → models/rag
   * YOLO disease model → models/yolo

(Models are loaded automatically during backend startup)

---

### ✅ 3. Dataset & Model Preparation 

* Custom agricultural QA dataset prepared from government sources
* Dataset cleaned and converted into instruction-based format
* QLoRA fine-tuning performed on Qwen 1.5 (1.8B)
* Final adapters saved and reused for inference


---

### ✅ 4. Run the Backend Server

1. Navigate to backend folder
2. Start FastAPI server using Uvicorn
3. Backend runs on local host (default port 8000)

Backend initializes:

* FAISS vector store
* Sentence embedding model
* Fine-tuned LLM
* YOLO disease detection model
* Speech & utility services

Once started, backend is ready to accept requests.

---

### ✅ 5. Run the Frontend (User Interface)

1. Navigate to frontend folder
2. Launch the Gradio application
3. Open the provided local URL in browser

The UI provides:

* Chat interface
* Voice assistant
* Disease detection upload
* Mandi prices
* Weather information

---

### ✅ 6. Using the Application

* Type or speak a farming question
* Upload a crop leaf image for disease detection
* Check mandi prices using crop and state
* Get real-time weather by location

All responses are generated using **trained models + real data pipelines**.

---

## 🔁 Project Pipeline (End-to-End Flow)

### 🧠 1. Data Pipeline

* Agricultural documents collected from government portals
* Cleaned, structured, and converted into QA format
* Stored for:

  * Fine-tuning
  * Retrieval (RAG)

---

### 📚 2. Training Pipeline

1. Base model: Qwen 1.5 (Chat)
2. Quantization: 4-bit (QLoRA)
3. LoRA adapters trained on custom dataset
4. Base model frozen, only adapters trained
5. Final adapters saved for inference

This enables efficient learning with limited GPU memory.

---

### 🔍 3. RAG Pipeline (Chat Queries)

1. User enters a question
2. Question converted into embedding
3. FAISS retrieves relevant documents
4. Retrieved context injected into prompt
5. Fine-tuned LLM generates grounded response

This reduces hallucination and improves factual accuracy.

---

### 🩺 4. Disease Detection Pipeline

1. User uploads crop leaf image
2. Image processed by YOLO model
3. Disease class predicted with confidence
4. Disease information generated as advisory

This connects **computer vision output to practical farming advice**.

---

### 🎙 5. Voice Interaction Pipeline

1. Farmer speaks a question
2. Speech converted to text
3. Text processed by chat pipeline
4. Answer converted back to speech
5. Audio reply returned to user

Useful for farmers with low literacy.

---

### 🌦 6. Utility Services Pipeline

* Mandi prices fetched from official APIs
* Weather data fetched in real-time
* Responses formatted for easy understanding

---

## 🧩 Overall Architecture Summary

User
→ Frontend (Gradio UI)
→ FastAPI Backend
→ LLM / RAG / Vision / APIs
→ Structured Farming Advice
→ Optional Voice Output

---

## ✅ Project Highlights for Evaluation

* Custom dataset creation
* QLoRA fine-tuning on real data
* RAG based knowledge grounding
* Vision + NLP integration
* Low-resource hardware optimization
* End-to-end deployable system

---


