## 🌾 KrishiGPT – AI Assistant for Farmers
#Overview

**KrishiGPT** is an end-to-end AI-powered agricultural assistant designed to help farmers with accurate, timely, and localized information.
It combines language models, voice interaction, computer vision, retrieval-based reasoning, and real-time data sources into a single unified system.

The platform supports:

Conversational AI for agriculture

Voice-based interaction

Crop disease detection from images

Weather forecasting

Nearby mandi (market) price information

Context-aware answers grounded in verified agricultural knowledge


KrishiGPT is built with a **production-style ML pipeline**, focusing on correctness, alignment, and real-world usability rather than just chatbot responses.


## 🎯 Core Objectives

* Reduce misinformation in agricultural guidance
* Provide **grounded, source-aware answers**
* Enable **low-literacy friendly access via voice**
* Assist farmers in **decision-making** (crop health, selling price, weather planning)

---

## ✨ Key Features

### 🤖 Conversational AI (Chat)

* Agriculture-focused question answering
* Context-aware and domain-aligned responses
* Prevents hallucinations using retrieval mechanisms

### 🎙️ Voice Chat (Speech-to-Text & Text-to-Speech)

* Farmers can speak queries instead of typing
* System converts speech → text → response → voice reply
* Designed for ease of use in rural environments

### 🌦️ Weather Information

* Real-time and forecast weather details
* Crop-relevant insights such as rainfall, temperature, humidity
* Helps farmers plan sowing, irrigation, and harvesting

### 📈 Nearby Mandi Price Rates

* Fetches local market (mandi) prices for crops
* Helps farmers decide **when and where to sell**
* Supports data-driven economic transparency

### 🌿 Crop Disease Detection (Computer Vision)

* Image-based crop disease identification
* Provides disease name and confidence
* Designed for quick on-field diagnosis

---

## 🏗️ System Architecture (High-Level)

```
User (Text / Voice / Image)
        ↓
Frontend (Gradio Interface)
        ↓
Decision Router
   ├─ Chat / RAG Module
   ├─ Voice Pipeline
   ├─ Weather Module
   ├─ Mandi Price Module
   └─ Vision (YOLO)
        ↓
Aggregated Response
        ↓
Text + Voice Output
```

---

## 🧠 AI & ML Pipeline (End-to-End)

### 1️⃣ Data Collection (Knowledge Source Creation)

* Agricultural manuals and PDFs
* Crop disease documentation
* Fertilizer and pesticide guidelines
* Government and institutional advisories

---

### 2️⃣ Data Cleaning & Preparation

* Removal of noise and duplicates
* Normalization of multilingual text
* Domain-specific formatting
* Chunking for semantic understanding

---

### 3️⃣ Embedding Generation

* Cleaned text converted into semantic embeddings
* Enables understanding of contextual similarity between queries and agricultural knowledge
* Forms the foundation of retrieval-based reasoning

---

### 4️⃣ Vector Store Creation

* Embeddings stored in a vector database
* Allows fast retrieval of relevant knowledge during inference
* Ensures responses are grounded in verified data

---

### 5️⃣ QA Dataset Preparation

* Automatic generation of:

  * High-quality correct answers
  * Incorrect or weaker answers (for contrast)
* Context-linked question–answer pairs
* Designed specifically for **preference learning**

---

## 🎯 Fine-Tuning Strategy

### Type of Fine-Tuning Used

✅ **Preference-Based Fine-Tuning**

Instead of traditional supervised fine-tuning, KrishiGPT uses **human-aligned preference learning**, where the model learns to choose:

* Better answers over poor or misleading ones
* Grounded responses over hallucinated responses

---

### Why Preference-Based Fine-Tuning?

* Improves factual reliability
* Reduces hallucination
* Aligns the model with real user expectations
* Especially effective for sensitive domains like agriculture

---

## 🔄 Retrieval-Augmented Generation (RAG)

During inference:

* User query is converted into embeddings
* Relevant agricultural context is retrieved
* Retrieved context is injected into the prompt
* Language model generates a grounded answer

✅ Ensures answers are **context-backed**, not guessed

---

## 🌿 Crop Disease Detection (YOLO)

### Model Overview

* Uses YOLO-based architecture for fast inference
* Trained on crop and leaf disease datasets
* Lightweight and suitable for real-time usage

### Purpose

* Identify plant diseases early
* Reduce dependency on manual expert visits
* Enable faster corrective action

---

## 🌦️ Weather Module

* Integrates external weather sources
* Provides farm-relevant insights
* Designed to support:

  * Irrigation planning
  * Pest risk assessment
  * Harvest timing decisions

---

## 📊 Mandi Price Intelligence

* Fetches crop prices from nearby mandis
* Reduces information asymmetry
* Helps farmers avoid underpricing exploitation
* Supports smarter selling strategies

---

## 🧪 Evaluation & Metrics

### 📌 Language Model Evaluation

* Human preference alignment
* Answer correctness
* Context faithfulness
* Response clarity and usefulness

### 📌 Vision Model Evaluation

* Precision
* Recall
* F1 Score
* Mean Average Precision (mAP)

---

## 🧑‍💻 Technology Stack

### AI / ML

* PyTorch
* Transformer-based Language Models
* Preference Optimization Techniques
* Retrieval-Augmented Generation (RAG)
* YOLO for Computer Vision

### Backend

* FastAPI
* Modular ML pipelines
* Vector databases for embeddings

### Frontend

* Gradio (interactive UI)
* Multimodal input support (text, voice, image)

### Deployment

* Hugging Face Spaces
* CPU-based inference
* Model and data caching for efficiency

---

## ⚠️ Limitations

* CPU-only inference on free hosting
* Large models require optimization
* Voice input limited to file-based audio on hosted platforms

---

## 🔮 Future Scope

* Multilingual Indian language support
* Soil health analysis
* Crop recommendation system
* Personalized farmer profiles
* Mobile application integration
* GPU-backed deployment for scale

---

## 👤 Author

**Ranjan Yadav**
AI / ML Engineer | Data Scientist

* GitHub: [https://github.com/ranjanr6](https://github.com/ranjanr6)
* Hugging Face: [https://huggingface.co/ranjanr6](https://huggingface.co/ranjanr6)

---

## 📜 License

MIT License – Open for use, modification, and extension.

---
