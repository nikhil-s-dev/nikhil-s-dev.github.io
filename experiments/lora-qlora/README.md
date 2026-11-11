# Expanded LoRA vs QLoRA comparison

Two matched seeds (42 and 43) compare LoRA and QLoRA on a Tesla T4, using Qwen/Qwen2.5-1.5B-Instruct and a fixed, shuffled Dolly 15k split: 1,000 training examples, 200 validation examples and five separate generation examples.

| Metric | LoRA seed 42 | QLoRA seed 42 | LoRA seed 43 | QLoRA seed 43 |
|---|---:|---:|---:|---:|
| Training time, including checkpoint saves | 385.21 s | 528.54 s | 405.01 s | 545.64 s |
| Training examples / second | 2.599 | 1.894 | 2.472 | 1.834 |
| Peak allocated GPU memory | 3.624 GiB | 2.714 GiB | 3.624 GiB | 2.715 GiB |
| Peak reserved GPU memory | 3.988 GiB | 3.068 GiB | 3.863 GiB | 3.068 GiB |
| Held-out language-model loss | 1.852 | 1.888 | 1.849 | 1.885 |

Averaging the two runs per method, QLoRA used about **25.1% less peak PyTorch allocated memory** (2.715 vs 3.624 GiB) and its training took about **35.9% longer** (537.09 vs 395.11 seconds). These are observations for this workload and two seeds; they are not general performance guarantees or confidence estimates. The matched model revision, T4, dataset split, and package versions were identical across all four runs.

Both methods used a maximum sequence length of 256, batch size 1, gradient accumulation 4, 250 optimizer steps, learning rate 2e-4, FP16 compute, and LoRA rank 16 / alpha 32. QLoRA used NF4 and double quantization. Peak values measure PyTorch memory during training, including allocations already present at the start, but exclude model-load peaks and non-PyTorch GPU allocations. Training time includes checkpoint writes; it excludes loading, validation, and generation. Validation loss covers instruction, context, and response tokens and is not an instruction-following quality score.

## Qualitative review

A manual, non-blinded review of five held-out prompts for each run is illustrative, not a representative quality evaluation. All four generated correct summaries of a provided hotel description, but all four invented the same incorrect founder for a fictional house and supplied fabricated details. Both QLoRA runs misclassified auto and umbrella insurance in a list task. One LoRA seed classified the list correctly, while the other initially classified it correctly but added contradictory explanations. Generated album lists also contained inaccuracies or unsupported claims. No independent pretrained-model baseline or human rating study was performed. Plain instruction/context/response prompts were used rather than the model's chat template; generation was capped at 96 new tokens and some answers were cut off. Reference answers may also need independent verification.

## Reproduce

The measured environment was Python 3.13.16, PyTorch 2.14.1+cu130, transformers 4.51.3, PEFT 0.15.2, datasets 3.5.0, accelerate 1.6.0, bitsandbytes 0.50.2. The model revision recorded by all runs was `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; the shuffled dataset fingerprint was `0c0b93c19c5b5a29`. The script records the resolved revision and fingerprint, but does not pin them on download: verify both before treating a rerun as matched.

Install a CUDA-compatible PyTorch for the environment, then install [requirements-expanded.txt](requirements-expanded.txt), and launch [expanded_benchmark.py](expanded_benchmark.py) in separate processes:

```bash
python expanded_benchmark.py lora 42
python expanded_benchmark.py qlora 42
python expanded_benchmark.py lora 43
python expanded_benchmark.py qlora 43
```

See the four raw metric records in [results](results/). The original measured Colab run saved checkpoints in temporary runtime storage, where they can disappear on reset. For a stronger follow-up, pin dataset and model revisions explicitly, repeat more seeds, track total process VRAM, add a pretrained baseline and blinded generation review, and test broader workloads.

# Earlier 32-step pilot


A small, single-seed experiment comparing adapter fine-tuning with an FP16 base model (LoRA) and 4-bit NF4 quantized base weights (QLoRA). Both runs used the same Colab Tesla T4, Qwen/Qwen2.5-1.5B-Instruct, Dolly 15k dataset, seed 42, 128 training examples, 32 held-out examples, maximum sequence length 256, batch size 1, gradient accumulation 4, 32 steps, learning rate 2e-4, and LoRA rank 16 / alpha 32. QLoRA used NF4, double quantization, and FP16 computation.

| Measurement | LoRA | QLoRA |
|---|---:|---:|
| Training wall time | 57.35 s | 73.19 s |
| Peak PyTorch allocated GPU memory during training | 3.624 GiB | 2.716 GiB |
| Peak PyTorch reserved GPU memory during training | 3.863 GiB | 3.068 GiB |
| Held-out language-model loss | 1.877 | 1.924 |
| Training loss | 2.052 | 2.112 |

These are observations from one short run, not universal performance claims. QLoRA used approximately 25.1% less peak allocated memory and took approximately 27.6% longer in this run. Measurements include model allocations present when training starts but exclude model-load peaks and non-PyTorch GPU allocations. Timings include the Trainer's training loop, not model download, loading, or evaluation. Evaluation loss includes prompt and response tokens; it is not a direct measure of instruction-following quality. No generation-based evaluation or repeated-seed confidence estimates were performed.

Environment: Python 3.13.16, PyTorch 2.14.1+cu130, transformers 4.51.3, peft 0.15.2, datasets 3.5.0, accelerate 1.6.0. bitsandbytes 0.45.5 was installed for the LoRA arm; the QLoRA arm required 0.50.2 for CUDA 13 support. The dataset was shuffled with seed 42 before selecting disjoint slices. The pilot used one run per method; package-version differences and runtime variability limit direct comparisons. No employer or proprietary data was used.

See [benchmark.py](benchmark.py) for the experiment code. Use a CUDA-enabled environment and install compatible PyTorch, transformers, PEFT, datasets, accelerate, and bitsandbytes versions. Run `python benchmark.py lora` and `python benchmark.py qlora` in separate processes. The script writes a JSON record for each method to the current directory. For a stronger study, pin model and dataset revisions, use matched package versions for both arms, repeat across seeds, track total GPU process memory, and evaluate generation quality on a separate test set.
