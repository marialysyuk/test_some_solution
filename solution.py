import json, sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from tqdm import tqdm

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DIR = "./llama3.1-8b-instruct-awq"
BATCH_SIZE = 8            # safer for unknown GPU; use 16 if you’re sure
MAX_NEW_TOKENS = 16
MAX_INPUT_LEN = 768       # safer than 1024 on small VRAM

SYSTEM_PROMPT = (
    "Отвечай ТОЛЬКО по-русски и ТОЛЬКО коротким фактом. "
    "Не перефразируй вопрос, в ответе должен быть ТОЛЬКО короткий ответ. "
    "Если тебе задали некорректный вопрос отвечай 'вопрос некорректен'. "
    "Если ты очень не уверена в ответе, отвечай 'не знаю'."
)

def build_batch(tokenizer, device, questions):
    ids = []
    for q in questions:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        x = tokenizer.apply_chat_template(
            msgs,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LEN,
        )
        ids.append(x[0])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    batch = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}

def decode_new_only(tokenizer, out_ids, in_ids):
    answers = []
    for i in range(out_ids.size(0)):
        gen = out_ids[i, in_ids[i].size(0):]
        txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
        # cut any leaked special prompts
        for stop in ("<|user|>", "<|system|>", "<|assistant|>"):
            pos = txt.find(stop)
            if pos != -1:
                txt = txt[:pos].strip()
        answers.append(txt)
    return answers

def read_questions(in_path="input.json"):
    p = Path(in_path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # fallback: try stdin if file absent
    data = sys.stdin.read().strip()
    if data:
        return json.loads(data)
    raise FileNotFoundError(f"input not found: {in_path} and stdin empty")

def main(in_path="input.json", out_path="output.json"):
    # make sure model dir exists (fail early with clear message)
    if not Path(MODEL_DIR).exists():
        print(f"ERROR: model dir not found: {MODEL_DIR}", file=sys.stderr)
        sys.exit(2)

    model = AutoAWQForCausalLM.from_quantized(
        MODEL_DIR, device_map="auto", fuse_layers=True,
        torch_dtype=torch.float16, trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False, trust_remote_code=True)

    # ensure pad ids on model too
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.eos_token_id

    device = next(model.parameters()).device

    questions = read_questions(in_path)
    if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
        raise ValueError("input must be a JSON list of strings")

    outputs = []
    model.eval()
    with torch.inference_mode():
        for i in tqdm(range(0, len(questions), BATCH_SIZE)):
            chunk = questions[i:i + BATCH_SIZE]
            batch = build_batch(tok, device, chunk)
            out = model.generate(
                **batch,
                do_sample=False, temperature=0.0, top_p=1.0,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
            )
            answers = decode_new_only(tok, out, batch["input_ids"])
            answers = [a.strip('«»"').rstrip('.').strip() or "не знаю" for a in answers]
            outputs.extend(answers)

    Path(out_path).write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} answers to {out_path}")

if __name__ == "__main__":
    main()
