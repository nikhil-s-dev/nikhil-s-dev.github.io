"""Expanded LoRA/QLoRA T4 comparison. Run once per method and seed."""
import sys, time, json, platform
from pathlib import Path
import torch, transformers, datasets, peft, bitsandbytes, accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

method, seed = sys.argv[1], int(sys.argv[2])
set_seed(seed)
assert method in ("lora", "qlora") and torch.cuda.is_available()
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
dataset_id = "databricks/databricks-dolly-15k"
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token
raw = load_dataset(dataset_id, split="train").shuffle(seed=2026).select(range(1205))
def tokenize(batch):
    texts = [f"Instruction: {a}\nContext: {b}\nResponse: {c}" + tok.eos_token for a, b, c in zip(batch["instruction"], batch["context"], batch["response"])]
    return tok(texts, truncation=True, max_length=256)
data = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
train, val = data.select(range(1000)), data.select(range(1000, 1200))
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16) if method == "qlora" else None
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, quantization_config=quant, device_map={"": 0})
revision = getattr(model.config, "_commit_hash", None)
model.config.use_cache = False
if method == "qlora":
    model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
model.enable_input_require_grads()
out = Path("results") / f"{method}-seed{seed}"
out.mkdir(parents=True, exist_ok=True)
args = TrainingArguments(output_dir=str(out), per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4, max_steps=250, learning_rate=2e-4, fp16=True, gradient_checkpointing=True, logging_steps=25, save_strategy="steps", save_steps=125, save_total_limit=2, report_to="none", seed=seed, data_seed=seed, disable_tqdm=True)
trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
result = trainer.train()
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
allocated = torch.cuda.max_memory_allocated() / 2**30
reserved = torch.cuda.max_memory_reserved() / 2**30
evaluation = trainer.evaluate()
model.eval()
model.config.use_cache = True
samples = []
for row in raw.select(range(1200, 1205)):
    prompt = f"Instruction: {row['instruction']}\nContext: {row['context']}\nResponse:"
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
    samples.append({"instruction": row["instruction"], "context": row["context"], "reference": row["response"], "generated": tok.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)})
metrics = {"method": method, "seed": seed, "model": model_id, "model_revision": revision, "dataset": dataset_id, "dataset_fingerprint": raw._fingerprint, "split_seed": 2026, "train_examples": 1000, "eval_examples": 200, "generation_examples": 5, "max_length": 256, "steps": 250, "batch_size": 1, "gradient_accumulation": 4, "learning_rate": 2e-4, "gpu": torch.cuda.get_device_name(0), "dtype": "float16", "train_seconds": elapsed, "train_examples_per_second": result.metrics.get("train_samples_per_second"), "peak_train_allocated_gib": allocated, "peak_train_reserved_gib": reserved, "train_loss": result.training_loss, "eval_loss": evaluation["eval_loss"], "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__}, "notes": "Training time includes checkpoint writes. Peak PyTorch memory excludes model-load peak and non-PyTorch VRAM. Eval loss covers prompt and response, not instruction quality."}
(out / "metrics.json").write_text(json.dumps(metrics, indent=2))
(out / "generation_samples.json").write_text(json.dumps(samples, indent=2))
print("METRICS " + json.dumps(metrics), flush=True)
print("SAMPLES " + json.dumps(samples), flush=True)
