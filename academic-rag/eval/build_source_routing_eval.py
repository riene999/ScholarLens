"""Build the 50-case multi-turn source-constraint routing benchmark."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


SOURCES = {
    "fedavg": "Communication-Efficient Learning of Deep Networks from Decentralized Data.pdf",
    "fedprox": "MLSys-2020-federated-optimization-in-heterogeneous-networks-Paper.pdf",
    "fedseq": "FedSeq_A_Hybrid_Federated_Learning_Framework_Based_on_Sequential_In-Cluster_Training.pdf",
    "spfl": "SPFL Sequential Updates with Parallel Aggregation for Enhanced Federated Learning Under Category and Domain Shifts.pdf",
    "tornado": "TornadoAggregate Accurate and Scalable Federated Learning.pdf",
    "dp": "Federated_Learning_With_Differential_Privacy_Algorithms_and_Performance_Analysis(1).pdf",
    "share": "SHARE Shaping Data Distribution at Edge for Communication-Efficient Hierarchical Federated Learning.pdf",
    "local_sgd": "Local SGD Converges Fast and Communicates Little.pdf",
    "periodic_sgd": "NeurIPS-2019-local-sgd-with-periodic-averaging-tighter-analysis-and-adaptive-synchronization-Paper.pdf",
    "diurnal": "Diurnal or nocturnal federated learning of multi-branch networks from periodically shifting distributions.pdf",
    "hfl": "A Joint Communication and Learning Framework for Hierarchical.pdf",
    "sequential": "Convergence Analysis of Sequential Federated Learning on Heterogeneous Data.pdf",
    "gda": "On Convergence of Gradient Descent Ascent A Tight Local Analysis.pdf",
    "reshuffle": "NeurIPS-2020-random-reshuffling-simple-analysis-with-vast-improvements-Paper.pdf",
    "noniid": "ON THE CONVERGENCE OF FEDAVG ON NON-IID.pdf",
    "resource": "Computation_and_Communication_Resource_Optimization_for_Efficient_Hierarchical_Federated_Learning.pdf",
}


def turn(sources: str | list[str], user: str, assistant: str) -> dict:
    keys = [sources] if isinstance(sources, str) else sources
    return {
        "turn_index": -1,
        "user": user,
        "assistant": assistant,
        "retrieved_sources": [SOURCES[key] for key in keys],
    }


def case(
    category: str,
    question: str,
    decision: str,
    *,
    sources: list[str] | None = None,
    recent_turns: list[dict] | None = None,
    current_source: str | None = None,
    basis: str = "none",
) -> dict:
    return {
        "category": category,
        "language": "zh" if any("\u4e00" <= char <= "\u9fff" for char in question) else "en",
        "question": question,
        "recent_turns": recent_turns or [],
        "current_source": SOURCES[current_source] if current_source else None,
        "expected_decision": decision,
        "expected_sources": [SOURCES[key] for key in (sources or [])],
        "expected_hard_scope": decision == "scoped",
        "expected_binding_basis": basis,
    }


CASES = [
    # 15 unique immediate follow-ups: these should be deterministic hard constraints.
    case("unique_follow_up", "那它的近端项系数怎么设置？", "scoped", sources=["fedprox"], recent_turns=[turn("fedprox", "FedProx 如何处理异构客户端？", "它通过近端项限制本地更新。")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "那这篇论文把通信轮数降低了多少？", "scoped", sources=["fedavg"], recent_turns=[turn("fedavg", "请介绍 Federated Averaging。", "论文提出在客户端执行多步本地训练后聚合。")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "这个方法为什么能减少上行通信？", "scoped", sources=["fedseq"], recent_turns=[turn("fedseq", "FedSeq 的核心训练流程是什么？", "客户端在簇内顺序训练，再上传聚合。")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "该论文把噪声加在什么位置？", "scoped", sources=["dp"], recent_turns=[turn("dp", "这项工作如何提供差分隐私？", "它提出了 NbAFL。")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "它具体在边缘侧塑造了什么？", "scoped", sources=["share"], recent_turns=[turn("share", "SHARE 为什么能降低云端通信？", "它在边缘侧重组数据分布。")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "What convergence guarantee does this paper prove?", "scoped", sources=["local_sgd"], recent_turns=[turn("local_sgd", "Summarize the Local SGD paper.", "It studies communication-efficient local updates.")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "What category-shift limitation does it address?", "scoped", sources=["spfl"], recent_turns=[turn("spfl", "What is SPFL?", "SPFL combines sequential updates with parallel aggregation.")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "How does that method control variance?", "scoped", sources=["tornado"], recent_turns=[turn("tornado", "Explain TornadoAggregate.", "It uses a ring-style hierarchical aggregation architecture.")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "它如何应对白天和夜晚的数据分布变化？", "scoped", sources=["diurnal"], recent_turns=[turn("diurnal", "介绍一下周期分布漂移的论文。", "论文采用多分支联邦模型。")], basis="recent_unique_antecedent"),
    case("unique_follow_up", "该方法解决了普通拆分联邦学习的哪些缺点？", "scoped", sources=["hfl"], recent_turns=[turn("hfl", "什么是分层拆分联邦学习？", "它在设备和边缘服务器之间拆分计算。")], basis="recent_unique_antecedent"),
    case("elliptical_follow_up", "在高度异构数据上得出了什么结论？", "scoped", sources=["sequential"], recent_turns=[turn("sequential", "顺序联邦学习如何更新客户端？", "客户端按顺序传递并更新模型。")], basis="recent_elliptical_follow_up"),
    case("elliptical_follow_up", "What stepsize ratio is required?", "scoped", sources=["gda"], recent_turns=[turn("gda", "What is the paper's local GDA result?", "It analyzes convergence near Stackelberg equilibria.")], basis="recent_elliptical_follow_up"),
    case("elliptical_follow_up", "Which restrictive assumptions are removed?", "scoped", sources=["reshuffle"], recent_turns=[turn("reshuffle", "Explain the Random Reshuffling result.", "It provides a simpler and tighter analysis.")], basis="recent_elliptical_follow_up"),
    case("elliptical_follow_up", "证明了什么收敛速度？", "scoped", sources=["noniid"], recent_turns=[turn("noniid", "FedAvg 在非 IID 数据上的主要理论结果是什么？", "论文分析了强凸光滑目标。")], basis="recent_elliptical_follow_up"),
    case("elliptical_follow_up", "联合优化了哪些资源变量？", "scoped", sources=["resource"], recent_turns=[turn("resource", "这篇分层联邦学习论文优化什么目标？", "它联合考虑计算和通信资源。")], basis="recent_elliptical_follow_up"),

    # 5 current-document references.
    case("current_document_reference", "这篇论文的核心贡献是什么？", "scoped", sources=["fedseq"], current_source="fedseq", basis="current_document_reference"),
    case("current_document_reference", "本文使用了哪些实验数据集？", "scoped", sources=["fedavg"], current_source="fedavg", basis="current_document_reference"),
    case("current_document_reference", "当前论文如何定义系统异构？", "scoped", sources=["fedprox"], current_source="fedprox", basis="current_document_reference"),
    case("current_document_reference", "What limitation does this paper report?", "scoped", sources=["spfl"], current_source="spfl", basis="current_document_reference"),
    case("current_document_reference", "What experiments does the current paper run?", "scoped", sources=["tornado"], current_source="tornado", basis="current_document_reference"),

    # 5 papers explicitly named in the current question.
    case("explicit_name", "根据 FedAvg 论文，它如何减少通信轮数？", "scoped", sources=["fedavg"], basis="explicit_name"),
    case("explicit_name", "在 FedProx 论文中，它如何同时处理系统异构和统计异构？", "scoped", sources=["fedprox"], basis="explicit_name"),
    case("explicit_name", "根据 FedSeq 论文，superclient 是怎么构造的？", "scoped", sources=["fedseq"], basis="explicit_name"),
    case("explicit_name", "SPFL 论文为什么要把顺序更新和并行聚合结合起来？", "scoped", sources=["spfl"], basis="explicit_name"),
    case("explicit_name", "According to the TornadoAggregate paper, what are its three design principles?", "scoped", sources=["tornado"], basis="explicit_name"),

    # 5 multi-paper references that require semantic/ordinal resolution.
    case("comparison_reference", "刚才两篇论文处理异构客户端的方式有什么不同？", "scoped", sources=["fedavg", "fedprox"], recent_turns=[turn(["fedavg", "fedprox"], "比较 FedAvg 和 FedProx。", "前者使用直接平均，后者加入近端项。")], basis="comparison_reference"),
    case("comparison_reference", "前者为什么更容易受到客户端漂移影响？", "scoped", sources=["fedavg"], recent_turns=[turn(["fedavg", "fedprox"], "FedAvg 和 FedProx 有什么区别？", "FedAvg 是前者，FedProx 是后者。")], basis="comparison_reference"),
    case("comparison_reference", "后者引入了哪个额外超参数？", "scoped", sources=["fedprox"], recent_turns=[turn(["fedavg", "fedprox"], "比较 FedAvg 和 FedProx。", "FedAvg 是前者，FedProx 是后者。")], basis="comparison_reference"),
    case("comparison_reference", "Compare the communication claims of the two papers.", "scoped", sources=["local_sgd", "periodic_sgd"], recent_turns=[turn(["local_sgd", "periodic_sgd"], "Compare these two Local SGD analyses.", "Both analyze periodic communication with different assumptions.")], basis="comparison_reference"),
    case("comparison_reference", "第一篇和第二篇分别如何组织顺序训练？", "scoped", sources=["fedseq", "spfl"], recent_turns=[turn(["fedseq", "spfl"], "比较 FedSeq 和 SPFL。", "第一篇是 FedSeq，第二篇是 SPFL。")], basis="comparison_reference"),

    # 15 global questions. The first ten deliberately include distracting context.
    case("global_with_context", "什么是联邦学习？", "global", recent_turns=[turn("fedavg", "请介绍 FedAvg。", "它是一种联邦优化方法。")]),
    case("global_with_context", "什么是非 IID 数据？", "global", current_source="fedseq"),
    case("global_with_context", "有哪些方法可以降低联邦学习通信成本？", "global", recent_turns=[turn("fedprox", "FedProx 做了什么？", "它处理异构问题。")]),
    case("global_with_context", "分层联邦学习通常有哪些系统组件？", "global", current_source="hfl"),
    case("global_with_context", "差分隐私的基本定义是什么？", "global", recent_turns=[turn("dp", "NbAFL 的结论是什么？", "它研究隐私与性能权衡。")]),
    case("global_with_context", "What is client drift in federated optimization?", "global", recent_turns=[turn("fedavg", "Explain FedAvg.", "It averages local model updates.")]),
    case("global_with_context", "What are common approaches to data heterogeneity?", "global", current_source="fedprox"),
    case("global_with_context", "How does split learning work in general?", "global", recent_turns=[turn("hfl", "Summarize hierarchical split FL.", "It distributes split computation across edge tiers.")]),
    case("global_with_context", "随机重排和普通 SGD 一般有什么区别？", "global", current_source="reshuffle"),
    case("global_with_context", "顺序训练有哪些常见优缺点？", "global", recent_turns=[turn("fedseq", "介绍 FedSeq。", "它采用簇内顺序训练。")]),
    case("standalone_global", "什么是模型聚合？", "global"),
    case("standalone_global", "How is communication complexity measured in distributed learning?", "global"),
    case("standalone_global", "联邦学习中常见的隐私攻击有哪些？", "global"),
    case("standalone_global", "What is the difference between convexity and the PL condition?", "global"),
    case("standalone_global", "如何评价一个 RAG 系统的召回质量？", "global"),

    # 5 genuinely ambiguous references: the router must not guess a paper.
    case("ambiguous_reference", "它的主要缺点是什么？", "ambiguous", recent_turns=[turn(["fedavg", "fedprox"], "比较 FedAvg 和 FedProx。", "两者都用于联邦优化。")]),
    case("ambiguous_reference", "What convergence rate does it prove?", "ambiguous", recent_turns=[turn(["local_sgd", "periodic_sgd"], "Discuss both Local SGD papers.", "They prove different rates.")]),
    case("ambiguous_reference", "这篇论文的结论是什么？", "ambiguous"),
    case("ambiguous_reference", "那它在实验中表现如何？", "ambiguous", recent_turns=[turn([], "我们刚才提到了两种方法。", "但没有检索具体论文。")]),
    case("ambiguous_reference", "该方法的理论假设是什么？", "ambiguous", recent_turns=[turn(["fedavg", "fedprox", "fedseq"], "概括这三种方法。", "三种方法的训练组织方式不同。")]),
]


def load_documents(database: Path) -> dict[str, dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, source_name, paper_title FROM documents"
        ).fetchall()
    finally:
        connection.close()
    return {str(row["source_name"]): dict(row) for row in rows}


def build(database: Path, output: Path) -> None:
    if len(CASES) != 50:
        raise RuntimeError(f"Expected 50 routing cases, found {len(CASES)}")
    documents = load_documents(database)
    records = []
    for index, item in enumerate(CASES, start=1):
        referenced_sources = list(item["expected_sources"])
        if item["current_source"]:
            referenced_sources.append(item["current_source"])
        for recent in item["recent_turns"]:
            referenced_sources.extend(recent["retrieved_sources"])
        missing = sorted(set(referenced_sources).difference(documents))
        if missing:
            raise RuntimeError(f"Routing case {index} references missing sources: {missing}")
        record = dict(item)
        current_source = record.pop("current_source")
        record["case_id"] = f"source_routing_{index:03d}"
        if current_source:
            document = documents[current_source]
            record["current_document"] = {
                "document_id": int(document["id"]),
                "source_name": current_source,
                "title": document["paper_title"],
            }
        else:
            record["current_document"] = None
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} source-routing cases to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/faiss_indexes/marker/documents.sqlite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/datasets/source_routing_eval_50.jsonl"),
    )
    args = parser.parse_args()
    build(args.database, args.output)


if __name__ == "__main__":
    main()
