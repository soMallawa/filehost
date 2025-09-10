import argparse, os, torch
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base HF model id or path (e.g., unsloth/gemma-3-4b-it)")
    ap.add_argument("--adapters", required=True, help="LoRA adapters folder (output of training)")
    ap.add_argument("--out", required=True, help="Output dir for merged fp16 weights")
    ap.add_argument("--gguf_dir_q4", default=None, help="Output dir for q4_k_m GGUF (optional)")
    ap.add_argument("--gguf_dir_f16", default=None, help="Output dir for f16 GGUF (optional)")
    ap.add_argument("--max_seq_length", type=int, default=4096)
    args = ap.parse_args()

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[INFO] Loading base in 16-bit (no 4bit) from {args.base}")
    base_model, tok = FastLanguageModel.from_pretrained(
        args.base,
        max_seq_length=args.max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=False,   # IMPORTANT: load full precision to allow merge
    )

    print(f"[INFO] Loading LoRA adapters from {args.adapters}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapters)

    print("[INFO] Merging LoRA into base (merge_and_unload) ...")
    merged = peft_model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    print(f"[INFO] Saving merged 16-bit weights to {args.out}")
    merged.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    # Optional GGUF exports via Unsloth (using a FastLanguageModel handle)
    if args.gguf_dir_q4 or args.gguf_dir_f16:
        print("[INFO] Reloading merged model with FastLanguageModel for GGUF export...")
        merged_model, merged_tok = FastLanguageModel.from_pretrained(
            args.out,
            max_seq_length=args.max_seq_length,
            dtype=torch_dtype,
            load_in_4bit=False,
        )

        if args.gguf_dir_q4:
            print(f"[INFO] Saving GGUF q4_k_m to {args.gguf_dir_q4}")
            merged_model.save_pretrained_gguf(args.gguf_dir_q4, merged_tok, quantization_method="q4_k_m")

        if args.gguf_dir_f16:
            print(f"[INFO] Saving GGUF f16 to {args.gguf_dir_f16}")
            merged_model.save_pretrained_gguf(args.gguf_dir_f16, merged_tok, quantization_method="f16")

    print("[INFO] Done.")

if __name__ == "__main__":
    main()