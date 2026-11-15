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

Open [the Colab notebook](vllm_gpu_benchmark.ipynb) to measure offline vLLM batch generation on an available GPU. It records results to a downloadable JSON file. It has not yet been run or validated in Colab, so GPU results are pending.

## Next measurement

Use [llm-loadgate](https://github.com/nikhil-s-dev/llm-loadgate) against the gateway and directly against the same model server. Record the model revision, server version, GPU, request lengths, load pattern, repetitions, TTFT, throughput, errors and cost assumptions. No GPU benchmark has been run for this project yet.

This is a local prototype: it has no client authentication, distributed rate limiting, metrics, TLS termination or production deployment. The timeout is a socket timeout, not an end-to-end deadline. It does not launch vLLM or Triton.
