# Retrieval-augmented generation evaluation report

- Corpus: **20** documents | Evaluation set: **40** questions
- Embedder: `tfidf` | Index: `faiss` | Judge: `local`
- Bootstrap resamples: 10,000 | Permutations: 10,000 | Significance level: 0.05

## Headline

- **answer_relevance**, `tfidf_baseline` vs `minilm`: no significant difference (delta=-0.0173; p=0.1660)
- **answer_relevance**, `minilm` vs `minilm_rerank`: no significant difference (delta=-0.0004; p=0.6260)

## Configurations

| Configuration | Chunk size | Overlap | Top-k | Embedder | Chunks indexed |
| --- | ---: | ---: | ---: | --- | ---: |
| `tfidf_baseline` | 128 | 20 | 5 | `tfidf(svd_dim=128)` | 60 |
| `minilm` | 128 | 20 | 5 | `sentence-transformers(model_name=sentence-transformers/all-MiniLM-L6-v2)` | 60 |
| `minilm_rerank` | 128 | 20 | 5 | `sentence-transformers(model_name=sentence-transformers/all-MiniLM-L6-v2) + rerank(cross-encoder/ms-marco-MiniLM-L-6-v2)` | 60 |

## Per-configuration metrics (mean with 95% bootstrap CI)

### `tfidf_baseline`

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

### `minilm`

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
| answer_relevance | 0.1860 | [0.1545, 0.2188] |
| context_utilization | 0.0903 | [0.0844, 0.0967] |

### `minilm_rerank`

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
| answer_relevance | 0.1863 | [0.1556, 0.2185] |
| context_utilization | 0.0859 | [0.0804, 0.0919] |

## Pairwise comparisons

Each pair is evaluated on the same questions. The difference is config A minus config B; the p-value comes from a two-sided paired permutation test.

### `tfidf_baseline` vs `minilm`

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
| answer_relevance | 0.1687 | 0.1860 | -0.0173 | [-0.0408, 0.0064] | 0.1660 | no significant difference (delta=-0.0173; p=0.1660) |
| context_utilization | 0.0845 | 0.0903 | -0.0059 | [-0.0137, 0.0018] | 0.1470 | no significant difference (delta=-0.0059; p=0.1470) |

### `minilm` vs `minilm_rerank`

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
| answer_relevance | 0.1860 | 0.1863 | -0.0004 | [-0.0018, 0.0009] | 0.6260 | no significant difference (delta=-0.0004; p=0.6260) |
| context_utilization | 0.0903 | 0.0859 | +0.0045 | [0.0022, 0.0068] | 0.0005 **\*** | minilm is significantly better (delta=+0.0045; p=0.0005) |

**\*** marks a statistically significant difference at the 0.05 level.
