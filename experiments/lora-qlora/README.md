# Expanded comparison: seed 42

The 250-step comparison uses 1,000 training and 200 validation examples with Qwen/Qwen2.5-1.5B-Instruct on a Tesla T4. Both arms used the same fixed split, seed, model revision and installed package versions, including bitsandbytes 0.50.2. A second seed is planned/in progress; these values describe only seed 42.

| Measurement | LoRA | QLoRA |
|---|---:|---:|
| Training wall time (includes checkpoint saves) | 385.21 s | 528.54 s |
| Training examples / second | 2.599 | 1.894 |
| Peak PyTorch allocated GPU memory | 3.624 GiB | 2.714 GiB |
| Peak PyTorch reserved GPU memory | 3.988 GiB | 3.068 GiB |
| Held-out language-model loss | 1.852 | 1.888 |

In this pair, QLoRA used about 25.1% less peak PyTorch allocated memory, but took about 37.2% longer. Both used 256 maximum tokens, batch size 1, gradient accumulation 4, rank 16, FP16 compute, and 250 optimization steps; QLoRA additionally used NF4 and double quantization. Peak memory excludes model load peak and non-PyTorch allocations; evaluation loss covers prompt and response and is not a quality score. The training time includes checkpoint writes. Results from a single pair are not sufficient to characterize variability.

A manual, non-blinded inspection of five held-out generations found that both models accurately summarized a provided hotel description but both invented an incorrect answer to a fictional-universe factual question. In the list-classification example, the LoRA response initially classified all items correctly but added contradictory explanations; QLoRA incorrectly excluded auto and umbrella insurance. These five examples are illustrative, not a statistically representative quality evaluation. Reference answers may themselves require independent verification.

Run [expanded_benchmark.py](expanded_benchmark.py) separately for each method and seed; raw records for completed runs are in [results](results/). In the measured Colab session, training checkpoints were in temporary runtime storage, so they can disappear when the runtime resets.

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
