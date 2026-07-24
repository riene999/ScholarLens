"""Evaluate retrieval, source routing, and optional answer generation.

Examples:
    python eval/evaluate_academic_rag.py --mode global
    python eval/evaluate_academic_rag.py --mode adaptive
    python eval/evaluate_academic_rag.py --mode adaptive --with-answers
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main as api
from src.rag.pipeline import RAGPipeline
from src.rag.source_router import SourceRouter
from src.utils.config import load_config


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON on line {line_number}: {exc}") from exc
    if not cases:
        raise RuntimeError("Evaluation dataset is empty")
    return cases


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return round(statistics.fmean(values), 6) if values else 0.0


def first_relevant_rank(values: list[str], gold: set[str]) -> int | None:
    return next((index for index, value in enumerate(values, start=1) if value in gold), None)


def reciprocal_rank(values: list[str], gold: set[str]) -> float:
    rank = first_relevant_rank(values, gold)
    return 1.0 / rank if rank else 0.0


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def source_ndcg_at_k(ranked_sources: list[str], gold_sources: set[str], k: int) -> float:
    unique_sources = unique_in_order(ranked_sources)[:k]
    dcg = sum(
        (1.0 if source in gold_sources else 0.0) / math.log2(rank + 1)
        for rank, source in enumerate(unique_sources, start=1)
    )
    ideal_relevant = min(len(gold_sources), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_relevant + 1))
    return dcg / idcg if idcg else 0.0


def score_case(case: dict, chunks: list, latency_ms: int, routing: dict | None) -> dict:
    ranked_sources = [str((chunk.document.metadata or {}).get("source") or "") for chunk in chunks]
    ranked_chunk_ids = [str(chunk.document.chunk_id) for chunk in chunks]
    gold_sources = set(case["gold_sources"])
    gold_chunks = {item["chunk_id"] for item in case["gold_evidence"]}
    found_sources = gold_sources.intersection(ranked_sources)
    result = {
        "case_id": case["case_id"],
        "question": case["question"],
        "case_type": case["case_type"],
        "topic": case["topic"],
        "difficulty": case["difficulty"],
        "gold_sources": case["gold_sources"],
        "gold_chunk_ids": sorted(gold_chunks),
        "ranked_sources": ranked_sources,
        "ranked_chunk_ids": ranked_chunk_ids,
        "scores": [round(float(chunk.score), 6) for chunk in chunks],
        "source_reciprocal_rank": reciprocal_rank(ranked_sources, gold_sources),
        "passage_reciprocal_rank": reciprocal_rank(ranked_chunk_ids, gold_chunks),
        "source_hit": bool(found_sources),
        "passage_hit": bool(gold_chunks.intersection(ranked_chunk_ids)),
        "source_recall": len(found_sources) / len(gold_sources),
        "source_precision": (
            sum(1 for source in ranked_sources if source in gold_sources) / len(ranked_sources)
            if ranked_sources
            else 0.0
        ),
        "complete_source_coverage": found_sources == gold_sources,
        "source_ndcg": source_ndcg_at_k(ranked_sources, gold_sources, len(ranked_sources)),
        "latency_ms": latency_ms,
        "routing": routing,
    }
    if routing and "decision" in routing:
        expected_route_sources = set(case.get("expected_route_sources", case["gold_sources"]))
        routed_sources = set(routing.get("source_names") or [])
        intersection = routed_sources.intersection(expected_route_sources)
        result["route_source_precision"] = (
            len(intersection) / len(routed_sources)
            if routed_sources
            else float(not expected_route_sources)
        )
        result["route_source_recall"] = (
            len(intersection) / len(expected_route_sources)
            if expected_route_sources
            else float(not routed_sources)
        )
        result["route_exact_match"] = routed_sources == expected_route_sources
        result["route_decision_correct"] = (
            routing.get("decision") == case.get("expected_route_decision")
            if case.get("expected_route_decision")
            else None
        )
    return result


def add_answer_scores(result: dict, case: dict, answer: str) -> None:
    lowered = answer.casefold()
    keyword_hits = [keyword for keyword in case["expected_keywords"] if keyword.casefold() in lowered]
    cited_sources = [source for source in case["gold_sources"] if source.casefold() in lowered]
    result.update({
        "answer": answer,
        "keyword_hits": keyword_hits,
        "keyword_recall": len(keyword_hits) / len(case["expected_keywords"]),
        "all_keywords_hit": len(keyword_hits) == len(case["expected_keywords"]),
        "gold_citation_recall": len(cited_sources) / len(case["gold_sources"]),
        "has_bracket_citation": "[" in answer and "]" in answer,
    })


def summarize(results: list[dict], top_k: int, with_answers: bool) -> dict:
    def block(items: list[dict]) -> dict:
        summary = {
            "case_count": len(items),
            f"source_mrr@{top_k}": mean(item["source_reciprocal_rank"] for item in items),
            f"passage_mrr@{top_k}": mean(item["passage_reciprocal_rank"] for item in items),
            f"source_hit_rate@{top_k}": mean(float(item["source_hit"]) for item in items),
            f"passage_hit_rate@{top_k}": mean(float(item["passage_hit"]) for item in items),
            f"mean_source_recall@{top_k}": mean(item["source_recall"] for item in items),
            f"mean_source_precision@{top_k}": mean(item["source_precision"] for item in items),
            f"complete_source_coverage@{top_k}": mean(
                float(item["complete_source_coverage"]) for item in items
            ),
            f"source_ndcg@{top_k}": mean(item["source_ndcg"] for item in items),
            "mean_latency_ms": mean(item["latency_ms"] for item in items),
        }
        if with_answers and items:
            summary.update({
                "mean_keyword_recall": mean(item["keyword_recall"] for item in items),
                "all_keywords_hit_rate": mean(float(item["all_keywords_hit"]) for item in items),
                "gold_citation_recall": mean(item["gold_citation_recall"] for item in items),
                "bracket_citation_rate": mean(float(item["has_bracket_citation"]) for item in items),
            })
        return summary

    single = [item for item in results if item["case_type"] == "single_source"]
    multi = [item for item in results if item["case_type"] == "multi_source"]
    summary = {
        "overall": block(results),
        "single_source": block(single),
        "multi_source": block(multi),
    }
    router_cases = [
        item
        for item in results
        if (item.get("routing") or {}).get("origin")
        in {"speculative_router", "validated_source_constraint"}
    ]
    routed = [item for item in router_cases if "route_source_recall" in item]
    if router_cases:
        summary["routing"] = {
            "total_cases": len(router_cases),
            "evaluated_cases": len(routed),
            "route_available_rate": len(routed) / len(router_cases),
            "scope_decision_rate": mean(
                float((item.get("routing") or {}).get("decision") == "scoped") for item in routed
            ),
            "route_source_precision": mean(item["route_source_precision"] for item in routed),
            "route_source_recall": mean(item["route_source_recall"] for item in routed),
            "route_exact_match_rate": mean(float(item["route_exact_match"]) for item in routed),
            "route_decision_accuracy": mean(
                float(item["route_decision_correct"])
                for item in routed
                if item.get("route_decision_correct") is not None
            ),
            "timeout_rate": mean(
                float(bool((item.get("routing") or {}).get("router_timed_out")))
                for item in router_cases
            ),
        }
    return summary


def write_report(
    output: Path,
    *,
    mode: str,
    top_k: int,
    dataset: Path,
    decomposition_enabled: bool,
    with_answers: bool,
    router_timeout_ms: int,
    router_grace_ms: int,
    results: list[dict],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": mode,
        "top_k": top_k,
        "dataset": str(dataset),
        "query_decomposition_enabled": decomposition_enabled,
        "answer_generation_enabled": with_answers,
        "router_timeout_ms": router_timeout_ms,
        "router_grace_ms": router_grace_ms,
        "metrics": summarize(results, top_k, with_answers),
        "cases": results,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


async def evaluate(args) -> None:
    cases = load_cases(args.dataset)
    config = load_config(args.config)
    if args.disable_decomposition:
        config.retrieval.query_decomposition_enabled = False
    if args.router_timeout_ms is not None:
        config.source_routing.total_timeout_ms = args.router_timeout_ms
    if args.router_grace_ms is not None:
        config.source_routing.grace_ms = args.router_grace_ms
    if args.with_answers and args.disable_citation_check:
        config.llm.citation_check_enabled = False

    pipeline = RAGPipeline(config)
    api.rag_pipeline = pipeline
    api.app_store = None
    api.source_router = SourceRouter(config.llm, config.source_routing)
    results = []

    for index, case_data in enumerate(cases, start=1):
        started = time.perf_counter()
        if args.mode == "adaptive":
            current_document = case_data.get("current_document") or {}
            chunks, routing = await api._retrieve_with_source_routing(
                case_data["question"],
                top_k=args.top_k,
                current_document_id=current_document.get("document_id"),
                current_source_name=current_document.get("source_name"),
                router_context=case_data.get("router_context"),
            )
        elif args.mode == "expanded":
            candidates = await asyncio.to_thread(
                pipeline.retrieve_candidates,
                case_data["question"],
                candidate_k=config.source_routing.global_candidate_k,
            )
            chunks = await asyncio.to_thread(
                pipeline.finalize_candidates,
                case_data["question"],
                candidates,
                top_k=args.top_k,
            )
            routing = {"mode": "global", "origin": "expanded_candidate_baseline"}
        elif args.mode == "oracle":
            chunks = await asyncio.to_thread(
                pipeline.retrieve_chunks,
                case_data["question"],
                top_k=args.top_k,
                source_filter=case_data["gold_sources"],
            )
            routing = {
                "mode": "hard",
                "origin": "gold_oracle",
                "source_names": case_data["gold_sources"],
            }
        else:
            chunks = await asyncio.to_thread(
                pipeline.retrieve_chunks,
                case_data["question"],
                top_k=args.top_k,
            )
            routing = {"mode": "global", "origin": "evaluation_baseline"}

        latency_ms = int((time.perf_counter() - started) * 1000)
        result = score_case(case_data, chunks, latency_ms, routing)
        if args.with_answers:
            answer = await asyncio.to_thread(
                pipeline.generator.generate,
                case_data["question"],
                chunks,
            )
            add_answer_scores(result, case_data, answer)
        results.append(result)
        write_report(
            args.output,
            mode=args.mode,
            top_k=args.top_k,
            dataset=args.dataset,
            decomposition_enabled=config.retrieval.query_decomposition_enabled,
            with_answers=args.with_answers,
            router_timeout_ms=config.source_routing.total_timeout_ms,
            router_grace_ms=config.source_routing.grace_ms,
            results=results,
        )
        logger.info(
            "Evaluation {}/{}: {} source_recall={:.2f}",
            index,
            len(cases),
            case_data["case_id"],
            result["source_recall"],
        )

    print(json.dumps(summarize(results, args.top_k, args.with_answers), indent=2))
    print(f"Full report: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/datasets/academic_eval_50_reviewed.jsonl"),
    )
    parser.add_argument(
        "--mode",
        choices=["global", "expanded", "adaptive", "oracle"],
        default="global",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--with-answers", action="store_true")
    parser.add_argument("--disable-decomposition", action="store_true")
    parser.add_argument("--disable-citation-check", action="store_true")
    parser.add_argument("--router-timeout-ms", type=int)
    parser.add_argument("--router-grace-ms", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        suffix = "_answers" if args.with_answers else ""
        args.output = Path(f"eval/results/results_{args.mode}_top{args.top_k}{suffix}.json")
    return args


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    asyncio.run(evaluate(parse_args()))
