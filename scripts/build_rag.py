import os, json, faiss, torch
import numpy as np
import pdfplumber
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
CSV_DIR = os.path.join(DATA_DIR, "csv")
OUT_DIR = os.path.join(ROOT, "models", "rag")
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 8
CHUNK = 400
OVERLAP = 50

def extract_pdf(path):
    text = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            t = p.extract_text()
            if t: text.append(t)
    return " ".join(text)

def load_csv(path):
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        line = " ".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
        out.append(line)
    return out

def clean(t): return " ".join(t.replace("\n"," ").split())

def chunk(t):
    words = t.split()
    out = []
    i=0
    while i < len(words):
        out.append(" ".join(words[i:i+CHUNK]))
        i += CHUNK - OVERLAP
    return out

def load_model(m):
    tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    model = AutoModel.from_pretrained(m, trust_remote_code=True).to(DEVICE)
    model.eval()
    return tokenizer, model

def embed(tokenizer, model, texts):
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (last * mask).sum(1) / mask.sum(1)
        emb = pooled.cpu().numpy()
        emb = normalize(emb)
    return emb

# Gather chunks
chunks = []
idx=0

for f in os.listdir(PDF_DIR):
    if f.endswith(".pdf"):
        text = clean(extract_pdf(os.path.join(PDF_DIR, f)))
        for c in chunk(text):
            chunks.append({"id": f"pdf_{idx}", "text": c, "source": f})
            idx+=1

for f in os.listdir(CSV_DIR):
    if f.endswith(".csv"):
        for row in load_csv(os.path.join(CSV_DIR, f)):
            for c in chunk(clean(row)):
                chunks.append({"id": f"csv_{idx}", "text": c, "source": f})
                idx+=1

# Load embedding model
try:
    tokenizer, model = load_model(PRIMARY_MODEL)
    used = PRIMARY_MODEL
except:
    tokenizer, model = load_model(FALLBACK_MODEL)
    used = FALLBACK_MODEL

# Embed chunks
texts = [c["text"] for c in chunks]
embs = []

for i in tqdm(range(0, len(texts), BATCH)):
    batch = texts[i:i+BATCH]
    embs.append(embed(tokenizer, model, batch))

embs = np.vstack(embs).astype("float32")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)

faiss.write_index(index, os.path.join(OUT_DIR, "faiss_index.idx"))
np.save(os.path.join(OUT_DIR, "embeddings.npy"), embs)

with open(os.path.join(OUT_DIR, "chunks.jsonl"), "w") as f:
    for c in chunks: f.write(json.dumps(c)+"\n")

with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump({"model": used}, f)

print("RAG build completed!")
