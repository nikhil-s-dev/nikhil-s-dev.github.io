"""Single-seed LoRA/QLoRA T4 pilot; run each method in a separate process."""
import sys, time, json, platform
from pathlib import Path
import torch, transformers, datasets, peft, bitsandbytes, accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if len(sys.argv) != 2 or sys.argv[1] not in ("lora", "qlora"):
    raise SystemExit("Usage: python benchmark.py {lora|qlora}")
method = sys.argv[1]
assert torch.cuda.is_available(), "A CUDA GPU is required"
seed = 42
set_seed(seed)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token
raw = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=seed).select(range(160))
def tokenize(batch):
    texts = [f"Instruction: {a}\nContext: {b}\nResponse: {c}" + tok.eos_token for a, b, c in zip(batch["instruction"], batch["context"], batch["response"])]
    return tok(texts, truncation=True, max_length=256)
data = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
train, val = data.select(range(128)), data.select(range(128, 160))
quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16) if method == "qlora" else None
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, quantization_config=quant, device_map={"": 0})
model.config.use_cache = False
if method == "qlora":
    model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
model.enable_input_require_grads()
args = TrainingArguments(output_dir="bench-" + method, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4, max_steps=32, learning_rate=2e-4, fp16=True, gradient_checkpointing=True, logging_steps=8, save_strategy="no", report_to="none", seed=seed, disable_tqdm=True)
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
metrics = {"method": method, "model": model_id, "dataset": "databricks/databricks-dolly-15k", "seed": seed, "train_examples": 128, "eval_examples": 32, "max_length": 256, "steps": 32, "batch_size": 1, "gradient_accumulation": 4, "gpu": torch.cuda.get_device_name(0), "dtype": "float16", "train_seconds": elapsed, "peak_train_allocated_gib": allocated, "peak_train_reserved_gib": reserved, "train_loss": result.training_loss, "eval_loss": evaluation["eval_loss"], "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "datasets": datasets.__version__, "peft": peft.__version__, "bitsandbytes": bitsandbytes.__version__, "accelerate": accelerate.__version__}, "notes": "Peak includes model already loaded; excludes model-load peak and non-PyTorch allocations. Eval loss includes prompt and response tokens. Single-seed pilot."}
Path(method + "-metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
