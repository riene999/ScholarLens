# Academic RAG evaluation

## Dataset

`datasets/academic_eval_50_reviewed.jsonl` contains 50 English questions derived from the
30 papers currently indexed in `data/faiss_indexes/marker`:

- 40 single-source questions, including one core-contribution question for every paper and 10 detailed result/theory questions.
- 10 multi-source comparison or synthesis questions involving two or three papers.
- Every case includes a reference answer, expected keywords, gold source files, gold chunk IDs, document IDs, chunk indexes, and a short evidence excerpt.

The source papers are English, so this first set uses English questions to measure
retrieval without mixing in cross-language embedding quality. A Chinese or bilingual
set should be reported separately.

The current Marker index does not retain PDF page numbers, so `page` is `null` and
exact passage evaluation uses stable chunk IDs. This is a known metadata gap rather
than missing evaluation annotation.

Regenerate the JSONL file after rebuilding the index:

```powershell
.\.venv\Scripts\python.exe eval\build_academic_eval.py
```

The builder fails if any reviewed gold chunk no longer exists.

### Preserved legacy dataset

`datasets/academic_eval_100.jsonl` is the earlier 100-question dataset. It contains
25 single-hop and 75 multi-hop questions over 28 papers. All referenced papers are
still present, so the file is intentionally preserved. Its gold evidence records only
source filenames rather than exact document and chunk IDs, so it is not directly
compatible with the evaluator below and should be treated as a legacy regression set.

### Multi-turn source-routing dataset

`datasets/source_routing_eval_50.jsonl` contains a separate 50-case benchmark for
source-constraint detection. It covers unique follow-ups, elliptical follow-ups,
current-document references, explicit paper names, multi-paper references, general
questions with distracting paper context, standalone general questions, and genuinely
ambiguous references. Keeping it separate prevents routing classification from being
mixed into passage-retrieval MRR.

Regenerate it with:

```powershell
.\.venv\Scripts\python.exe eval\build_source_routing_eval.py
```

## Metrics

- **Source MRR@K**: reciprocal rank of the first retrieved chunk whose source paper is a gold source.
- **Passage MRR@K**: reciprocal rank of the first exact gold chunk.
- **Source/Passage Hit@K**: whether at least one gold source or exact gold chunk appears.
- **Mean Source Recall@K**: fraction of required papers represented in the retrieved chunks, averaged over questions.
- **Complete Source Coverage@K**: fraction of questions for which every required paper is represented. This is the main multi-source metric.
- **Source Precision@K**: fraction of returned chunks coming from a gold source.
- **Source nDCG@K**: rank-sensitive gain after deduplicating sources in first-seen order.
- **Route availability/precision/recall/exact match**: whether the asynchronous route arrived in time and whether its document set matches the gold set.
- **Optional answer metrics**: expected-keyword recall and citation-source recall when `--with-answers` is enabled. These lexical metrics are diagnostic and should not be presented as a complete answer-quality score.

## Reproducible commands

Production-style baseline with query decomposition:

```powershell
.\.venv\Scripts\python.exe eval\evaluate_academic_rag.py `
  --mode global --top-k 5 `
  --output eval\results\results_global_top5.json
```

Deterministic retrieval comparison without LLM query decomposition:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'

.\.venv\Scripts\python.exe eval\evaluate_academic_rag.py `
  --mode global --top-k 5 --disable-decomposition `
  --output eval\results\results_global_no_decomposition_top5.json

.\.venv\Scripts\python.exe eval\evaluate_academic_rag.py `
  --mode expanded --top-k 5 --disable-decomposition `
  --output eval\results\results_expanded_no_decomposition_top5.json
```

Adaptive routing parameter experiment:

```powershell
.\.venv\Scripts\python.exe eval\evaluate_academic_rag.py `
  --mode adaptive --top-k 5 `
  --router-timeout-ms 4000 --router-grace-ms 1000 `
  --output eval\results\results_adaptive_safe_4s_grace1s_top5.json
```

Evaluate the complete source-constraint router, including the five cases that require
the configured LLM:

```powershell
.\.venv\Scripts\python.exe eval\evaluate_source_router.py --timeout-ms 15000 `
  --output eval\results\source_routing_eval_50.json
```

This command sends the question, recent-turn source metadata, and document catalog to
the LLM endpoint configured in `config.yaml`.

## Results from 2026-07-17

### Deterministic retrieval ablation

Query decomposition was disabled for both runs. Model loading is excluded from
per-case latency, but the expanded run benefited from warm OS/GPU caches, so latency
should not be compared between these two single runs.

| Mode | Overall source MRR@5 | Passage MRR@5 | Passage Hit@5 | Multi-source mean recall@5 | Multi-source complete coverage@5 |
|---|---:|---:|---:|---:|---:|
| Direct Top-5 | 0.923 | 0.594 | 80% | 68.3% | 40% |
| Expanded global candidates, no routing | 0.935 | 0.641 | 92% | 71.7% | 40% |

The broader candidate pool improves exact-passage retrieval, but does not solve the
problem of representing every required paper in a five-passage context.

### Earlier relevance-based routing experiments

| Mode | Source MRR@5 | Source Hit@5 | Multi-source mean recall@5 | Multi-source complete coverage@5 | Mean latency |
|---|---:|---:|---:|---:|---:|
| Global retrieval + query decomposition | 0.920 | 100% | 83.3% | 70% | 1.96 s |
| Historical 1 s router timeout / 200 ms grace | 0.945 | 100% | 83.3% | 70% | 2.08 s |
| Safe soft routing, 4 s timeout / 1 s grace | 0.950 | 96% | 88.3% | 80% | 3.26 s |

These runs predate the source-constraint redesign and are retained as failure-analysis
evidence, not as current router claims. Important interpretation:

- With the historical 1-second total timeout, the route was available for 0/50 cases. The observed difference from the global baseline therefore came from the broader 40-candidate pool, not document routing.
- Increasing the wait budget makes routing available more often, but LLM latency and route variability are substantial. In an earlier strong-routing run, 37/50 routes arrived, yet multi-source complete coverage fell from 70% to 40% and latency rose to 3.35 seconds.
- These results showed that asking an LLM to select topically relevant papers was the wrong task: general questions could be over-constrained, and an unverified route could remove useful evidence.
- The final safe-routing run recovered multi-source complete coverage to 80%, but route availability was only 46% in that network run and single-source hit rate was 95%. These are useful engineering results, but not stable enough for a resume claim until repeated over at least three runs with a faster router and corrected document titles.

### Source-constraint routing gate

The redesigned router and the pre-change title/filename keyword matcher were evaluated
on the same 50 multi-turn cases. The redesigned run used the configured DeepSeek
endpoint and was a complete run, not a local simulation:

| Metric | Legacy keyword matcher | Redesigned router |
|---|---:|---:|
| Total decision accuracy | 15/50 (30%) | 50/50 (100%) |
| Constraint recall | 0% | 100% |
| Exact source match on scoped cases | 0% | 30/30 (100%) |
| Ambiguous-reference recall | 0% | 100% |
| False scope rate on general questions | 0% | 0% |

Execution details for the redesigned router:

| Metric | Result |
|---|---:|
| Cases resolved locally | 45/50 (90%) |
| Cases executed by the LLM | 5/50 (10%) |
| LLM decision accuracy | 5/5 (100%) |
| Timeout / error rate | 0% / 0% |
| Calls within the 5 s router timeout | 50/50 (100%) |
| End-to-end routing latency P50 / P95 | 1 ms / 2707 ms |

The five LLM cases are elliptical follow-ups with no explicit paper name or pronoun,
such as asking “What stepsize ratio is required?” immediately after a single-paper
turn. Their measured latencies were 2123–2943 ms. The old 1-second production timeout
would therefore have discarded valid routes; `config.yaml` now allows a 5-second total
timeout and waits up to 4 seconds after concurrent retrieval. The full per-case report
is stored in `results/source_routing_eval_50.json`.

## Recommended resume-safe numbers

The most defensible current retrieval statement remains:

> Built a 50-question, 30-paper evaluation set with exact source and passage annotations; two-stage candidate expansion and reranking raised passage MRR@5 from 0.594 to 0.641 and passage Hit@5 from 80% to 92%, while multi-source passage Hit@5 increased from 70% to 90%.

The separate routing result may be stated as a 50-case source-constraint classification
test with 100% decision accuracy and 0% false scope on general questions. It should not
be described as improving final-answer quality or multi-source coverage because this
benchmark tests routing decisions, not answer generation.
