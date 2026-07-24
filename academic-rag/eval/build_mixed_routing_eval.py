"""Build one end-to-end retrieval set mixing scoped and global questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _continue_sentence(prefix: str, question: str) -> str:
    if not question:
        return prefix
    return f"{prefix}{question[0].lower()}{question[1:]}"


def build(source: Path, output: Path) -> None:
    cases = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cases) != 50:
        raise RuntimeError(f"Expected 50 reviewed cases, found {len(cases)}")

    all_sources = list(dict.fromkeys(
        source_name
        for case in cases
        for source_name in case["gold_sources"]
    ))
    mixed = []
    for index, original in enumerate(cases):
        case = dict(original)
        case["original_question"] = original["question"]
        scoped = index % 2 == 0
        if scoped:
            plural = len(case["gold_sources"]) > 1
            noun = "papers" if plural else "paper"
            case["question"] = _continue_sentence(
                f"Using only the {noun} discussed in the previous turn, ",
                original["question"],
            )
            previous_user = f"Please introduce the {noun} we will use next."
            previous_assistant = (
                f"I have retrieved the relevant {noun} and will use "
                f"{'them' if plural else 'it'} for the next question."
            )
            case["router_context"] = {
                "conversation_summary": "",
                "recent_messages": [
                    {"role": "user", "content": previous_user},
                    {"role": "assistant", "content": previous_assistant},
                ],
                "recent_turns": [{
                    "turn_index": -1,
                    "user": previous_user,
                    "assistant": previous_assistant,
                    "retrieved_sources": case["gold_sources"],
                    "artifact_digests": [],
                }],
                "user_profile": "",
            }
            case["current_document"] = None
            case["expected_route_decision"] = "scoped"
            case["expected_route_sources"] = case["gold_sources"]
        else:
            distractor = next(
                source_name
                for source_name in all_sources
                if source_name not in set(case["gold_sources"])
            )
            case["question"] = _continue_sentence(
                "Across the full paper collection, ",
                original["question"],
            )
            case["router_context"] = {
                "conversation_summary": "",
                "recent_messages": [],
                "recent_turns": [],
                "user_profile": "",
            }
            case["current_document"] = {"source_name": distractor}
            case["expected_route_decision"] = "global"
            case["expected_route_sources"] = []
        mixed.append(case)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in mixed) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("eval/datasets/academic_eval_50_reviewed.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/datasets/academic_eval_50_mixed_routing.jsonl"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.source, args.output)
