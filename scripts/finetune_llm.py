import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# -------------------------
# PATH CONFIG
# -------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))

# ✅ USE CLEAN DATASET (VERY IMPORTANT)
DATA_FILE = os.path.join(ROOT, "data", "finetune_dataset_clean.jsonl")

OUTPUT_DIR = os.path.join(ROOT, "models", "llm", "adapter")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# BASE MODEL
# -------------------------
BASE_MODEL = "Qwen/Qwen1.5-1.8B-Chat"

print(f"🔹 Loading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# -------------------------
# 4-BIT QLoRA CONFIG (GPU SAFE – 4GB)
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print("🔹 Loading base model in 4-bit QLoRA mode...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

# -------------------------
# LoRA CONFIG
# -------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("🔹 Applying LoRA adapters...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# LOAD DATASET
# -------------------------
print("🔹 Loading QA fine-tuning dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def format_prompt(example):
    text = (
        "### Instruction:\n"
        f"{example['instruction']}\n\n"
        "### Question:\n"
        f"{example.get('input', '')}\n\n"
        "### Answer:\n"
        f"{example['output']}"
    )
    return {"text": text}

dataset = dataset.map(format_prompt)

print("🔹 Tokenizing dataset...")
dataset = dataset.map(
    lambda x: tokenizer(
        x["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    ),
    batched=True,
)

dataset = dataset.remove_columns(
    [c for c in dataset.column_names if c not in ["input_ids", "attention_mask"]]
)

# -------------------------
# DATA COLLATOR
# -------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# -------------------------
# ✅ FAST TRAINING ARGUMENTS (KEY FIX)
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,   # ✅ faster than 16
    num_train_epochs=2,              # ✅ enough (no 8 hr drama)
    learning_rate=2e-4,

    fp16=True,
    bf16=False,

    optim="paged_adamw_8bit",        # ✅ BIGGEST SPEED FIX
    logging_steps=20,

    save_strategy="epoch",
    save_total_limit=1,

    report_to="none",
)

# -------------------------
# TRAINER
# -------------------------
print("🔥 Starting QLoRA fine-tuning (GPU)…")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

# -------------------------
# SAVE ADAPTER
# -------------------------
print("💾 Saving LoRA adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ QLoRA fine-tuning completed successfully!")
