# Grounded RAG prototype

A local, extractive question-answering prototype over four fictional maintenance notes. It filters documents by a supplied group before BM25 ranking and returns source text with citations. The extractive response makes it possible to inspect the evidence without an LLM or network access.

```bash
python rag.py
# In a second terminal:
curl http://127.0.0.1:8090/ask -H 'Content-Type: application/json' \
  -d '{"query":"permit energy isolation","groups":["public"]}'
python -m unittest -v
python evaluate.py
```

In one local smoke run, all eight hand-written questions returned their labelled document at rank 1. See [the results](results-local.json) and [design notes](DESIGN.md). The questions were written alongside the four fictional documents, so this is a functionality check, not evidence of production accuracy. Local timings are specific to the run and should not be treated as service latency.

The `groups` value is supplied by the caller **for demonstration only**. It is not authentication: anyone can choose `maintenance`. Never expose this prototype or use it with sensitive information. In a real service, derive entitlements from a verified identity and enforce them at every retrieval layer.

The index is lexical BM25, not hybrid search, embeddings or a reranker. There is no LLM, ingestion pipeline, distributed storage or production deployment. Four fictional documents are not a benchmark. A matching document is not necessarily a correct answer. Evaluate retrieval separately with [rag-regress](https://github.com/nikhil-s-dev/rag-regress) using independently labelled questions; evaluate answer faithfulness and authorization on their own test sets.
