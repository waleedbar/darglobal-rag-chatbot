Retrieval evaluation - 41 labelled questions (28 EN / 13 AR), embedder `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, chunks 853.
| Index | Retriever | hit@1 | hit@5 | MRR@10 | hit@5 EN | hit@5 AR |
|---|---|---|---|---|---|---|
| built index | hybrid | 0.85 | 1.00 | 0.92 | 1.00 | 1.00 |
| built index | dense | 0.78 | 1.00 | 0.88 | 1.00 | 1.00 |
| built index | bm25 | 0.73 | 0.95 | 0.82 | 1.00 | 0.85 |
| 500-char chunks | hybrid | 0.83 | 1.00 | 0.90 | 1.00 | 1.00 |
| 500-char chunks | dense | 0.76 | 1.00 | 0.86 | 1.00 | 1.00 |
| 500-char chunks | bm25 | 0.73 | 0.95 | 0.82 | 1.00 | 0.85 |
| 1000-char chunks | hybrid | 0.80 | 0.95 | 0.88 | 0.96 | 0.92 |
| 1000-char chunks | dense | 0.80 | 0.95 | 0.87 | 0.96 | 0.92 |
| 1000-char chunks | bm25 | 0.80 | 0.93 | 0.86 | 0.96 | 0.85 |
Hybrid misses (not in top 5): none
