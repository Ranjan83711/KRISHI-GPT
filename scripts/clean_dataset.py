import json

IN = "../data/finetune_dataset.jsonl"
OUT = "../data/finetune_dataset_clean.jsonl"

with open(IN, "r", encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as o:
    for line in f:
        item = json.loads(line)

        # remove numbering like "1:", "2:" etc
        item["input"] = item["input"].lstrip("0123456789: ").strip()
        item["output"] = item["output"].lstrip("0123456789: ").strip()

        o.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Clean dataset created:", OUT)
