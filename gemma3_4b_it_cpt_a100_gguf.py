# -*- coding: utf-8 -*-
"""
CPT-only for Gemma-3 4B IT — Optimized for A100 SXM with GGUF export
---------------------------------------------------------------------
- Base: unsloth/gemma-3-4b-it (instruction-tuned)
- Task: Continued pretraining (domain adaptation) on Sinhala Wikipedia
- Precision: bf16 on A100 (if supported); 4-bit loading + LoRA
- Batch: per-device 8, grad-accum 2 (global ≈ 16)
- TRAIN_FRACTION controls corpus size (default 0.25 = 25%)
- Saves LoRA adapters, merged FP16 weights, and GGUF quantizations (q4_k_m + f16)
"""

import os, torch
from datasets import load_dataset
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments

# ------------------------
# Environment / defaults
# ------------------------
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")  # Disable CCE for CPT

DTYPE = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
DEVICE = "cuda"

MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/gemma-3-12b-it")  # IT checkpoint
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "4096"))
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "true").lower() in ("1","true","yes")

# Batch / steps (optimized for A100 SXM 80GB)
PER_DEVICE_BS = int(os.getenv("PER_DEVICE_BS", "8"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))
MAX_STEPS_CPT = int(os.getenv("MAX_STEPS_CPT", "120"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))

# LRs
LR = float(os.getenv("LR", "5e-5"))
EMB_LR = float(os.getenv("EMB_LR", "1e-5"))

# Data fraction for CPT
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.50"))  # 25% by default

# Optimizer and LoRA
OPTIM = os.getenv("OPTIM", "adamw_torch_fused")
LORA_R = int(os.getenv("LORA_R", "128"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.0"))
USE_RSLORA = os.getenv("USE_RSLORA", "true").lower() in ("1","true","yes")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs_a100_cpt_only")
SAVE_DIR = os.getenv("SAVE_DIR", "gemma3_4b_it_cpt_a100")
SAVE_MERGED_16BIT = os.getenv("SAVE_MERGED_16BIT", "true").lower() in ("1","true","yes")

# ------------------------
# Load model + LoRA
# ------------------------
print(f"[INFO] Loading base model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# For CPT on an IT model, training embeddings + lm_head is okay for domain/style shift.
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_R,
    target_modules = [
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
        "embed_tokens","lm_head",
    ],
    lora_alpha = LORA_ALPHA,
    lora_dropout = LORA_DROPOUT,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = USE_RSLORA,
    loftq_config = None,
)

# ------------------------
# Dataset — Sinhala Wikipedia CPT
# ------------------------
EOS = tokenizer.eos_token or "</s>"

def format_si_wiki(batch):
    titles = batch["title"]
    texts  = batch["text"]
    out = []
    for t, x in zip(titles, texts):
        out.append(f"විකිපීඩියා ලිපිය\n### මාතෘකාව: {t}\n\n### ලිපිය:\n{x}{EOS}")
    return {"text": out}

print("[INFO] Loading Sinhala Wikipedia (wikimedia/wikipedia 20231101.si)")
ds = load_dataset("wikimedia/wikipedia", "20231101.si", split="train")

if TRAIN_FRACTION < 1.0:
    ds = ds.train_test_split(train_size=TRAIN_FRACTION, seed=3407)["train"]
    print(f"[INFO] Using train fraction = {TRAIN_FRACTION} of corpus.")

ds = ds.map(format_si_wiki, batched=True, num_proc=8)

# ------------------------
# Trainer — CPT only
# ------------------------
args = UnslothTrainingArguments(
    per_device_train_batch_size = PER_DEVICE_BS,
    gradient_accumulation_steps = GRAD_ACCUM,
    max_steps = MAX_STEPS_CPT,
    warmup_ratio = WARMUP_RATIO,
    learning_rate = LR,
    embedding_learning_rate = EMB_LR,
    logging_steps = 1,
    optim = OPTIM,
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = f"{OUTPUT_DIR}/cpt",
    report_to = "none",
)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 8,
    args = args,
)

# GPU info
gpu = torch.cuda.get_device_properties(0)
print(f"[INFO] GPU: {gpu.name} | VRAM: {round(gpu.total_memory/1024/1024/1024,2)} GB")
start_reserved = round(torch.cuda.max_memory_reserved()/1024/1024/1024, 3)
print(f"[INFO] Start reserved: {start_reserved} GB")

print("==== Starting CPT (A100, CPT-only) ====")
stats = trainer.train()
print(f"[INFO] CPT runtime (s): {stats.metrics.get('train_runtime','NA')}")

# ------------------------
# Save artifacts (LoRA, merged, GGUF)
# ------------------------
from unsloth import FastLanguageModel as FLM
FLM.for_inference(model)

print("[INFO] Saving LoRA adapters...")
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

if SAVE_MERGED_16BIT:
    print("[INFO] Saving merged 16-bit weights (vLLM-friendly)...")
    model.save_pretrained_merged(f"{SAVE_DIR}_merged_fp16", tokenizer, save_method="merged_16bit")

# GGUF exports
print("[INFO] Saving GGUF q4_k_m...")
model.save_pretrained_gguf(f"{SAVE_DIR}_gguf_q4km", tokenizer, quantization_method="q4_k_m")

print("[INFO] Saving GGUF f16...")
model.save_pretrained_gguf(f"{SAVE_DIR}_gguf_f16", tokenizer, quantization_method="f16")

print("[INFO] Done. CPT-only finetune complete with GGUF exports.")
