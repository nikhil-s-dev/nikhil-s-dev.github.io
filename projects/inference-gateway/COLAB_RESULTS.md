# Offline vLLM batch experiment — November 15, 2026

This Colab run used a Tesla T4 and Qwen/Qwen2.5-0.5B-Instruct, with vLLM 0.33.0, PyTorch 2.14.0+cu130, Python 3.13.16 and CUDA 13.0. The notebook resolved the model without pinning a revision; a subsequent cache inspection showed snapshot `7ae557604adf67be50417f59c2c2f167def9a775`. Future runs should pin this revision explicitly.

| Batch size | Trial 1 | Trial 2 | Trial 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| 1 | 28.56 | 31.30 | 34.43 | 31.30 |
| 4 | 77.65 | 141.11 | 137.12 | 137.12 |
| 8 | 222.57 | 260.89 | 280.71 | 260.89 |

Values are aggregate output tokens per second, calculated from completed output tokens divided by the wall time of each offline `LLM.generate` call. Each request generated 64 tokens; the settings used temperature zero, `ignore_eos=True`, FP16, a 1,024-token context and eager execution. One batch of size one was used as warmup.

The first batch of four was notably slower than the other two. Runs were performed in ascending batch-size order using similar prompts; cache effects and order effects were not controlled. Three trials are insufficient to quantify uncertainty. The run does not measure gateway overhead, streaming time to first token, network traffic, GPU utilization, cost or production workload behavior. Package installation reported conflicts with preinstalled Colab libraries, although this experiment completed.

[Raw output](results/colab-t4-2026-11-15.json) · [Notebook](vllm_gpu_benchmark.ipynb)
