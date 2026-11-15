# Controlled T4 batch and serving tests

Run on November 15, 2026 in Google Colab on one Tesla T4, Python 3.13.16, PyTorch 2.14.0+cu130 and vLLM 0.33.0. The model was Qwen/Qwen2.5-0.5B-Instruct pinned to revision `7ae557604adf67be50417f59c2c2f167def9a775`. Eager FP16 execution and disabled prefix caching were used throughout. The gateway code was taken from commit `b57e068fff4be667d7e84bb2199fec288db81cbb`.

## Offline generation

Seven measured calls per batch size, each with the same prompts, 64 forced output tokens per request, one warmup per batch size and a seeded shuffled order. Rates are aggregate output tokens per second.

| Batch size | Median | Minimum | Maximum |
| --- | ---: | ---: | ---: |
| 1 | 36.08 | 31.30 | 37.60 |
| 4 | 142.61 | 126.85 | 147.38 |
| 8 | 286.52 | 231.53 | 290.42 |

## Streaming HTTP serving

The client, vLLM server, and gateway shared the same Colab VM. Direct calls to vLLM were compared with calls through the gateway. For each path and concurrency, three closed-loop runs sent 12 requests each. Each used the same prompt, 32 forced output tokens, and temperature zero; there were eight warmup requests per path. Group order was shuffled with a fixed seed. Prefix caching was disabled. Gateway admission limit was eight in-flight requests.

| Path | Concurrency | TTFT p50 | TTFT p95 | Output tokens/s | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 1 | 60.9 ms | 87.6 ms | 33.5 | 0/36 |
| Gateway | 1 | 61.8 ms | 87.1 ms | 33.7 | 0/36 |
| Direct | 4 | 100.2 ms | 132.3 ms | 112.1 | 0/36 |
| Gateway | 4 | 93.3 ms | 120.6 ms | 134.4 | 0/36 |
| Direct | 8 | 100.0 ms | 104.2 ms | 200.2 | 0/36 |
| Gateway | 8 | 106.3 ms | 121.4 ms | 197.7 | 0/36 |

TTFT is time from the start of a client request to the first nonempty content event. Percentiles pool 36 successful requests for each path/concurrency, with nearest-rank p95; token throughput is the median of three per-run aggregate rates. The output token count comes from the final server usage event. [Machine-readable summary](results/colab-t4-controlled-summary-2026-11-15.json).

## Interpretation and limits

All 216 measured requests returned HTTP 200, streamed content, a completion usage count, and the SSE end marker. Error-free results on this small run are not a reliability guarantee. At concurrency four the direct path varied from 109.8–135.0 tokens/s and the gateway from 132.7–135.5 tokens/s. The measured differences do **not** establish that the gateway speeds up the model: the trials were short, sequential groups on a shared VM, not matched or independently replicated deployments.

Thirty-six requests per group give a noisy p95. The load is closed-loop, not an externally fixed arrival rate; there was no load above the gateway limit of eight, no GPU utilization or cost measurement, no TLS, no separate client host, and no network or distributed deployment. CPU contention and Colab scheduling affect both client and server. The pip install reported conflicts with some preinstalled packages even though this run completed. Model outputs were not graded for quality.

The [original per-request results](results/colab-t4-controlled-2026-11-15.json) include all 21 offline trials and 216 measured serving requests. The [controlled benchmark notebook](controlled-benchmark.ipynb) contains the experiment code and a final server cleanup cell. The experiment's individual cells ran in Colab; the recovered notebook has not yet been rerun end-to-end from a fresh runtime.
