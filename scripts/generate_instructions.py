import os, json, random

ROOT = os.path.dirname(os.path.dirname(__file__))
CHUNKS = os.path.join(ROOT, "models", "rag", "chunks.jsonl")
OUT = os.path.join(ROOT, "data", "finetune_dataset.jsonl")

templates = [
    "Explain this in simple farmer language with steps.",
    "Summarize the disease symptoms and treatment clearly.",
    "Convert into a question-answer format for farmers.",
    "Give treatment, dosage, prevention steps.",
    "Rewrite this as an advisory message for a farmer."
]

chunks = [json.loads(l) for l in open(CHUNKS, "r")]

with open(OUT, "w", encoding="utf-8") as f:
    for c in chunks:
        t = random.choice(templates)
        item = {
            "instruction": t,
            "input": "",
            "output": c["text"]
        }
        f.write(json.dumps(item)+"\n")

print("Instruction dataset created:", OUT)
