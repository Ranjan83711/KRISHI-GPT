import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

ROOT = os.path.dirname(os.path.dirname(__file__))
CHUNKS = os.path.join(ROOT, "models", "rag", "chunks.jsonl")
OUT = os.path.join(ROOT, "data", "finetune_dataset.jsonl")

PROMPT = """
You are an agriculture expert helping Indian farmers.

From the given agriculture text:
1. Create 2 realistic farmer questions.
2. Give clear, practical answers in simple Hindi.
3. Avoid scientific language.

TEXT:
{chunk}

Format:
Q1: ...
A1: ...
Q2: ...
A2: ...
"""

with open(CHUNKS, "r", encoding="utf-8") as f:
    chunks = [json.loads(l)["text"] for l in f]

out = open(OUT, "w", encoding="utf-8")

for chunk in chunks[:2000]:   # ✅ pehle 2000 enough
    prompt = PROMPT.format(chunk=chunk[:2000])

    resp = client.chat.completions.create(
        model="gpt-4o-mini",   # 💸 cheap
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    text = resp.choices[0].message.content.strip()

    qa_blocks = text.split("Q")
    for b in qa_blocks[1:]:
        try:
            q, a = b.split("A", 1)
            item = {
                "instruction": "Answer the farmer question clearly.",
                "input": q.strip(": \n"),
                "output": a.strip()
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
        except:
            continue

out.close()
print("✅ Correct QA finetune dataset created:", OUT)
