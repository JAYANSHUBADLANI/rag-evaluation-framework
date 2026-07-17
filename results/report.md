# Retrieval-augmented generation evaluation report

- Corpus: **20** documents | Evaluation set: **40** questions
- Embedder: `tfidf` | Index: `faiss` | Judge: `local`
- Bootstrap resamples: 10,000 | Permutations: 10,000 | Significance level: 0.05

## Headline

- **context_utilization**, `baseline` vs `large_chunks`: baseline is significantly better (delta=+0.0299; p=0.0001)
- **context_utilization**, `baseline` vs `low_dim`: no significant difference (delta=-0.0046; p=0.2298)

## Configurations

| Configuration | Chunk size | Overlap | Top-k | Embedder | Chunks indexed |
| --- | ---: | ---: | ---: | --- | ---: |
| `baseline` | 128 | 20 | 5 | `tfidf(svd_dim=128)` | 60 |
| `large_chunks` | 256 | 20 | 5 | `tfidf(svd_dim=128)` | 40 |
| `low_dim` | 128 | 20 | 5 | `tfidf(svd_dim=8)` | 60 |

## Per-configuration metrics (mean with 95% bootstrap CI)

### `baseline`

| Metric | Mean | 95% CI |
| --- | ---: | :---: |
| precision@1 | 1.0000 | [1.0000, 1.0000] |
| recall@1 | 0.9750 | [0.9375, 1.0000] |
| ndcg@1 | 1.0000 | [1.0000, 1.0000] |
| hit_rate@1 | 1.0000 | [1.0000, 1.0000] |
| precision@3 | 0.3500 | [0.3333, 0.3750] |
| recall@3 | 1.0000 | [1.0000, 1.0000] |
| ndcg@3 | 1.0000 | [1.0000, 1.0000] |
| hit_rate@3 | 1.0000 | [1.0000, 1.0000] |
| precision@5 | 0.2100 | [0.2000, 0.2250] |
| recall@5 | 1.0000 | [1.0000, 1.0000] |
| ndcg@5 | 1.0000 | [1.0000, 1.0000] |
| hit_rate@5 | 1.0000 | [1.0000, 1.0000] |
| mrr | 1.0000 | [1.0000, 1.0000] |
| faithfulness | 1.0000 | [1.0000, 1.0000] |
| answer_relevance | 0.1687 | [0.1381, 0.2005] |
| context_utilization | 0.0845 | [0.0799, 0.0894] |

### `large_chunks`

| Metric | Mean | 95% CI |
| --- | ---: | :---: |
| precision@1 | 1.0000 | [1.0000, 1.0000] |
| recall@1 | 0.9750 | [0.9375, 1.0000] |
| ndcg@1 | 1.0000 | [1.0000, 1.0000] |
| hit_rate@1 | 1.0000 | [1.0000, 1.0000] |
| precision@3 | 0.3500 | [0.3333, 0.3750] |
| recall@3 | 1.0000 | [1.0000, 1.0000] |
| ndcg@3 | 1.0000 | [1.0000, 1.0000] |
| hit_rate@3 | 1.0000 | [1.0000, 1.0000] |
| precision@5 | 0.2100 | [0.2000, 0.2250] |
| recall@5 | 1.0000 | [1.0000, 1.0000] |
| ndcg@5 | 1.0000 | [1.0000, 1.0000] |
| hit_rate@5 | 1.0000 | [1.0000, 1.0000] |
| mrr | 1.0000 | [1.0000, 1.0000] |
| faithfulness | 1.0000 | [1.0000, 1.0000] |
| answer_relevance | 0.1829 | [0.1518, 0.2147] |
| context_utilization | 0.0546 | [0.0475, 0.0650] |

### `low_dim`

| Metric | Mean | 95% CI |
| --- | ---: | :---: |
| precision@1 | 0.9000 | [0.8000, 0.9750] |
| recall@1 | 0.8750 | [0.7750, 0.9625] |
| ndcg@1 | 0.9000 | [0.8000, 0.9750] |
| hit_rate@1 | 0.9000 | [0.8000, 0.9750] |
| precision@3 | 0.3417 | [0.3167, 0.3750] |
| recall@3 | 0.9750 | [0.9250, 1.0000] |
| ndcg@3 | 0.9473 | [0.8815, 0.9908] |
| hit_rate@3 | 0.9750 | [0.9250, 1.0000] |
| precision@5 | 0.2050 | [0.1900, 0.2250] |
| recall@5 | 0.9750 | [0.9250, 1.0000] |
| ndcg@5 | 0.9473 | [0.8815, 0.9908] |
| hit_rate@5 | 0.9750 | [0.9250, 1.0000] |
| mrr | 0.9375 | [0.8625, 0.9875] |
| faithfulness | 1.0000 | [1.0000, 1.0000] |
| answer_relevance | 0.1239 | [0.0929, 0.1562] |
| context_utilization | 0.0891 | [0.0833, 0.0953] |

## Pairwise comparisons

Each pair is evaluated on the same questions. The difference is config A minus config B; the p-value comes from a two-sided paired permutation test.

### `baseline` vs `large_chunks`

| Metric | Mean A | Mean B | Δ (A−B) | 95% CI of Δ | p-value | Verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| precision@1 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| recall@1 | 0.9750 | 0.9750 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| ndcg@1 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| hit_rate@1 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| precision@3 | 0.3500 | 0.3500 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| recall@3 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| ndcg@3 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| hit_rate@3 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| precision@5 | 0.2100 | 0.2100 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| recall@5 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| ndcg@5 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| hit_rate@5 | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| mrr | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| faithfulness | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| answer_relevance | 0.1687 | 0.1829 | -0.0143 | [-0.0393, 0.0115] | 0.2801 | no significant difference (delta=-0.0143; p=0.2801) |
| context_utilization | 0.0845 | 0.0546 | +0.0299 | [0.0194, 0.0380] | 0.0001 **\*** | baseline is significantly better (delta=+0.0299; p=0.0001) |

### `baseline` vs `low_dim`

| Metric | Mean A | Mean B | Δ (A−B) | 95% CI of Δ | p-value | Verdict |
| --- | ---: | ---: | ---: | :---: | ---: | --- |
| precision@1 | 1.0000 | 0.9000 | +0.1000 | [0.0250, 0.2000] | 0.1247 | no significant difference (delta=+0.1000; p=0.1247) |
| recall@1 | 0.9750 | 0.8750 | +0.1000 | [0.0250, 0.2000] | 0.1247 | no significant difference (delta=+0.1000; p=0.1247) |
| ndcg@1 | 1.0000 | 0.9000 | +0.1000 | [0.0250, 0.2000] | 0.1247 | no significant difference (delta=+0.1000; p=0.1247) |
| hit_rate@1 | 1.0000 | 0.9000 | +0.1000 | [0.0250, 0.2000] | 0.1247 | no significant difference (delta=+0.1000; p=0.1247) |
| precision@3 | 0.3500 | 0.3417 | +0.0083 | [0.0000, 0.0250] | 1.0000 | no significant difference (delta=+0.0083; p=1.0000) |
| recall@3 | 1.0000 | 0.9750 | +0.0250 | [0.0000, 0.0750] | 1.0000 | no significant difference (delta=+0.0250; p=1.0000) |
| ndcg@3 | 1.0000 | 0.9473 | +0.0527 | [0.0092, 0.1185] | 0.1247 | no significant difference (delta=+0.0527; p=0.1247) |
| hit_rate@3 | 1.0000 | 0.9750 | +0.0250 | [0.0000, 0.0750] | 1.0000 | no significant difference (delta=+0.0250; p=1.0000) |
| precision@5 | 0.2100 | 0.2050 | +0.0050 | [0.0000, 0.0150] | 1.0000 | no significant difference (delta=+0.0050; p=1.0000) |
| recall@5 | 1.0000 | 0.9750 | +0.0250 | [0.0000, 0.0750] | 1.0000 | no significant difference (delta=+0.0250; p=1.0000) |
| ndcg@5 | 1.0000 | 0.9473 | +0.0527 | [0.0092, 0.1185] | 0.1247 | no significant difference (delta=+0.0527; p=0.1247) |
| hit_rate@5 | 1.0000 | 0.9750 | +0.0250 | [0.0000, 0.0750] | 1.0000 | no significant difference (delta=+0.0250; p=1.0000) |
| mrr | 1.0000 | 0.9375 | +0.0625 | [0.0125, 0.1375] | 0.1247 | no significant difference (delta=+0.0625; p=0.1247) |
| faithfulness | 1.0000 | 1.0000 | +0.0000 | [0.0000, 0.0000] | 1.0000 | no significant difference (delta=+0.0000; p=1.0000) |
| answer_relevance | 0.1687 | 0.1239 | +0.0447 | [0.0190, 0.0716] | 0.0015 **\*** | baseline is significantly better (delta=+0.0447; p=0.0015) |
| context_utilization | 0.0845 | 0.0891 | -0.0046 | [-0.0122, 0.0028] | 0.2298 | no significant difference (delta=-0.0046; p=0.2298) |

**\*** marks a statistically significant difference at the 0.05 level.
