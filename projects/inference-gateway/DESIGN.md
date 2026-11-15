# Design notes

The client sends an OpenAI-compatible chat request to a local gateway. A bounded semaphore rejects excess in-flight requests with HTTP 429. The gateway forwards the request to a configured upstream and copies response bytes as they arrive. It closes the upstream connection and releases the slot in a `finally` block, including after an error or a client disconnect.

## What is measured

The included tests exercise streaming response forwarding, invalid input, the health endpoint and an unavailable upstream using a local mock. They do not demonstrate GPU throughput or production reliability. The optional Colab notebook measures **offline vLLM batch generation**, not gateway latency, streaming TTFT or Triton. Its JSON output records the hardware and software used.

For a network benchmark, run the same model and fixed workload directly against a server and through the gateway using `llm-loadgate`, with warmup and repeated trials. Report offered load, concurrency, p50/p95 TTFT, end-to-end latency, completed requests, error rate, token throughput, prompt and output distributions, GPU model and all software versions. Change one variable at a time. A mock server demonstrates the test setup only, not GPU performance.

## Interview discussion

- Bounded concurrency protects the upstream from unlimited in-flight work, but a local semaphore cannot enforce a limit across replicas. This prototype sheds load rather than queueing it.
- Socket timeouts bound inactive reads; they do not implement a total deadline. Retries can duplicate work and should not be added blindly to streaming requests.
- The gateway forwards stream bytes rather than interpreting token counts. A stream can fail after HTTP headers were sent; the client must detect incomplete responses.
- Production deployment needs authenticated clients, TLS, rate limits per tenant, observability, distributed admission control, safe configuration and load testing. These are not implemented here.
