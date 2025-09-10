import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base = "unsloth/gemma-3-4b-it"
adapters = "gemma3_4b_it_cpt_a100"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
).cuda().eval()

model = PeftModel.from_pretrained(base_model, adapters).eval()

def gen(p, max_new_tokens=256):
    ids = tok(p, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9)
    print(tok.decode(out[0], skip_special_tokens=True))

gen("සිංහලෙන් ශ්‍රී ලංකාවේ පුරාවිද්‍යාමය සම්පත් ගැන සාරාංශයක්.")
