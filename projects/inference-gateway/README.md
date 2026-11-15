# Local LLM inference gateway

A small HTTP gateway for a vLLM or other OpenAI-compatible chat-completions endpoint. It limits concurrent requests, forwards streamed responses and exposes a local health endpoint. It is a starting point for the GPU-serving project in my portfolio.

## Run

Requires Python 3.10+. Start a model server separately, then:

```bash
python gateway.py --upstream http://127.0.0.1:8000 --port 8080 --max-inflight 8
curl http://127.0.0.1:8080/health
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"YOUR_MODEL","messages":[{"role":"user","content":"Hello"}],"stream":true}'
python -m unittest -v
```

To pass a service credential to the upstream, set `UPSTREAM_API_KEY` in the gateway's environment. The gateway listens only on localhost and does not forward client authorization headers. Overload returns HTTP 429; an unavailable upstream returns 502. An upstream connection failure after streaming starts terminates the stream.

## GPU measurement

A Colab T4 run measured offline vLLM generation on Qwen2.5-0.5B-Instruct at batch sizes 1, 4 and 8. Three-trial medians were 31.30, 137.12 and 260.89 aggregate output tokens/s, respectively. See the [run details](COLAB_RESULTS.md), [raw results](results/colab-t4-2026-11-15.json), and [notebook](vllm_gpu_benchmark.ipynb). This is an offline model benchmark, not a test of the gateway or production latency.

A later [controlled offline and HTTP serving experiment](CONTROLLED_RESULTS.md) ran seven trials per batch and 216 measured streaming requests across direct and gateway paths. The [original measurements](results/colab-t4-controlled-2026-11-15.json) and [recovered benchmark notebook](controlled-benchmark.ipynb) are included. The notebook's cells were executed during the experiment, but the recovered notebook has not been rerun from start to finish in a fresh runtime.

## Next measurement

Use [llm-loadgate](https://github.com/nikhil-s-dev/llm-loadgate) against the gateway and directly against the same model server. Record the model revision, server version, GPU, request lengths, load pattern, repetitions, TTFT, throughput, errors and cost assumptions.

This is a local prototype: it has no client authentication, distributed rate limiting, metrics, TLS termination or production deployment. The timeout is a socket timeout, not an end-to-end deadline. It does not launch vLLM or Triton.
