# Design notes

## Data path

An incoming question is tokenized; the index first restricts its candidate documents to the caller's supplied groups, then computes BM25 scores over that visible subset. The highest-ranked passage is returned verbatim with its document ID. No model generates an answer. Empty results explicitly abstain.

Filtering before scoring matters: ranking across all documents and filtering afterward can reveal information through statistics or exhaust the top-k with forbidden results. This sample's caller-provided group is only a simulation; real entitlements must come from verified identity claims, be checked by the serving tier and data store, and be tested for tenant isolation.

## Evaluation and limits

`python evaluate.py` measures hit@1, hit@3 and local single-query timings for eight hand-written cases. The documents and questions were written together, so this is a smoke test with substantial selection bias. Four documents and eight queries are far too small to claim a general retrieval-quality improvement. Test authorization separately from relevance; build an independently labelled, representative held-out set before comparing systems.

This prototype has no chunking strategy, vector database, hybrid retrieval, reranking, ingestion job, durable index, LLM, answer-faithfulness evaluation, verified identity or audit trail. It is deliberately limited to a transparent reference implementation. A production design would also need document provenance, deletion and reindexing, prompt-injection defenses, monitoring, evaluation gates and lifecycle management.

## Interview discussion

- Why BM25? It is a fast, inspectable baseline; it can miss paraphrases and synonyms. Compare it with a semantic retriever on the same independently labelled queries.
- Why extractive answers? The response can be traced directly to a source, but an irrelevant passage can still be returned and it cannot synthesize multiple sources.
- Why separately test authorization? Good recall does not demonstrate tenant isolation. Verify that unauthorized content and scores cannot reach a caller under valid and adversarial requests.
