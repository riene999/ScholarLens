"""Build the reviewed 50-question academic RAG evaluation set.

The questions and reference answers are manually derived from the indexed
papers. This script resolves the reviewed evidence chunk IDs against the
current SQLite document store so stale or misspelled evidence cannot silently
enter the dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


def case(
    question: str,
    answer: str,
    keywords: list[str],
    chunks: list[str],
    *,
    case_type: str = "single_source",
    difficulty: str = "medium",
    topic: str,
) -> dict:
    return {
        "question": question,
        "expected_answer": answer,
        "expected_keywords": keywords,
        "gold_chunk_ids": chunks,
        "case_type": case_type,
        "difficulty": difficulty,
        "topic": topic,
    }


CASES = [
    case(
        "How does hierarchical split federated learning address the weaknesses of ordinary split federated learning?",
        "It groups devices, splits model computation between devices and edge servers, and performs edge aggregation. This reduces the single point of failure and improves fairness and convergence while jointly considering device association and transmit power.",
        ["grouping", "edge aggregation", "single point of failure", "transmit power"],
        ["A Joint Communication and Learning Framework for Hierarchical_chunk_0"],
        topic="hierarchical_fl",
    ),
    case(
        "What idea lets the unified permutation-SGD analysis cover dependent permutations such as GraB?",
        "The paper introduces a general assumption that explicitly captures dependencies between permutations across epochs, allowing arbitrary, independent, one-permutation, and dependent-permutation methods to share one analysis.",
        ["inter-epoch", "dependent permutations", "general assumption", "GraB"],
        ["A Unified Analysis of Stochastic Gradient Descent_chunk_0"],
        topic="optimization_theory",
    ),
    case(
        "What decisions are optimized in the communication-cost minimization formulation for hierarchical federated learning?",
        "The formulation jointly selects edge aggregators and assigns computing nodes to them, while shaping balanced edge data distributions so a target accuracy can be reached with less edge and cloud communication.",
        ["edge aggregator selection", "node-edge association", "communication cost", "data distribution"],
        ["A_Communication-Efficient_Hierarchical_Federated_Learning_Framework_via_Shaping_Data_Distribution_at_Edge_chunk_0"],
        topic="hierarchical_fl",
    ),
    case(
        "Why was Federated Averaging proposed, and how much can it reduce communication rounds compared with synchronized SGD?",
        "Federated Averaging keeps raw data on devices and averages locally computed model updates. The reported experiments reduce the required communication rounds by roughly 10 to 100 times compared with synchronized SGD.",
        ["locally-computed updates", "model averaging", "10", "100"],
        ["Communication-Efficient Learning of Deep Networks from Decentralized Data_chunk_0"],
        difficulty="easy",
        topic="federated_optimization",
    ),
    case(
        "What objective is optimized in the joint computation and communication resource method for hierarchical federated learning?",
        "It minimizes a weighted combination of federated-learning completion time and total energy consumption. The problem is split into four subproblems, using fractional programming for the first three and CCCP for the last.",
        ["completion time", "energy consumption", "fractional programming", "CCCP"],
        ["Computation_and_Communication_Resource_Optimization_for_Efficient_Hierarchical_Federated_Learning_chunk_0"],
        topic="resource_optimization",
    ),
    case(
        "What does the convergence analysis conclude about sequential versus parallel federated learning on highly heterogeneous data?",
        "It establishes guarantees for strongly convex, general convex, and non-convex objectives and concludes that sequential federated learning can outperform parallel federated learning when data heterogeneity is high, with either full or partial participation.",
        ["sequential", "parallel", "heterogeneous", "partial participation"],
        ["Convergence Analysis of Sequential Federated Learning on Heterogeneous Data_chunk_0"],
        topic="sequential_fl",
    ),
    case(
        "How does the proposed method handle client distributions that shift between daytime and nighttime modes?",
        "It models the shift as a gradual mixture of daytime and nighttime distributions, jointly learns a clustering model and a multi-branch network, assigns lightweight specialized branches, and uses a temporal prior.",
        ["mixture", "multi-branch", "temporal prior", "daytime"],
        ["Diurnal or nocturnal federated learning of multi-branch networks from periodically shifting distributions_chunk_0"],
        topic="distribution_shift",
    ),
    case(
        "Which two game models are used for self-organizing edge association and resource allocation in hierarchical federated learning?",
        "Worker edge-association behavior is modeled with an evolutionary game at the lower level, while resource allocation between the model owner and edge servers is modeled with a Stackelberg differential game at the upper level.",
        ["evolutionary game", "Stackelberg differential game", "edge association", "resource allocation"],
        ["Dynamic Edge Association and Resource Allocation in Self-Organizing Hierarchical Federated Learning Networks_chunk_0"],
        topic="resource_optimization",
    ),
    case(
        "Which costs are compared when evaluating federated learning for future 5G and 6G edge networks?",
        "The study evaluates alternative federated-learning approaches in terms of training time, communication overhead, and energy consumption under resource-constrained edge-network conditions.",
        ["training time", "communication overhead", "energy consumption", "6G"],
        ["Efficient training Federated learning cost analysis_chunk_0"],
        topic="systems_evaluation",
    ),
    case(
        "How do federated learning and split learning compare under imbalanced and extremely non-IID IoT data?",
        "On the tested Raspberry Pi setting, split learning performs better under imbalanced data, but federated learning performs better under extremely non-IID data. The paper also evaluates splitfed learning as a combination of both approaches.",
        ["split learning", "imbalanced", "extreme non-IID", "Raspberry Pi"],
        ["Evaluation and optimization of distributed machine learning techniques for internet of things_chunk_0"],
        topic="systems_evaluation",
    ),
    case(
        "Where is noise added in NbAFL, and what privacy-performance trade-off does its analysis reveal?",
        "NbAFL adds artificial noise to client parameters before server aggregation. Stronger privacy requires more noise and worsens convergence, while relaxing privacy improves the trained model's loss.",
        ["before aggregation", "artificial noise", "privacy", "convergence"],
        ["Federated_Learning_With_Differential_Privacy_Algorithms_and_Performance_Analysis(1)_chunk_0"],
        topic="privacy",
    ),
    case(
        "How does FedSeq reduce uplink communication while improving accuracy on non-IID data?",
        "FedSeq clusters users, lets only a cluster head upload to the parameter server, and trains sequentially inside each cluster so the model sees more data categories and receives more updates per epoch.",
        ["cluster head", "sequential in-cluster", "uplink", "non-IID"],
        ["FedSeq_A_Hybrid_Federated_Learning_Framework_Based_on_Sequential_In-Cluster_Training_chunk_0"],
        topic="sequential_fl",
    ),
    case(
        "What three topology-selection principles are reported for hierarchical federated learning under data heterogeneity?",
        "Top-tier aggregation affects convergence more than intra-group aggregation; Ring-Star works well with many small groups while Star-Ring suits fewer dense groups; and inter-group heterogeneity is the dominant convergence bottleneck.",
        ["top-tier", "Ring-Star", "Star-Ring", "inter-group heterogeneity"],
        ["iclr_2026_chunk_0"],
        topic="hierarchical_fl",
    ),
    case(
        "What makes the convergence guarantees for sequential federated learning in Sharp Bounds 'sharp'?",
        "The work gives upper bounds for strongly convex, general convex, and non-convex objectives and matching lower bounds for the convex cases, showing that the rates cannot generally be improved and that sequential training benefits from high heterogeneity.",
        ["upper bounds", "lower bounds", "strongly convex", "heterogeneity"],
        ["jmlr_reply_chunk_0"],
        topic="sequential_fl",
    ),
    case(
        "How should user association differ between IID and non-IID wireless hierarchical federated learning?",
        "For IID data, devices can prefer the base station with the best uplink SNR to reduce latency. For non-IID data, association must jointly consider uplink SNR and how the device's data distribution affects distribution distance.",
        ["uplink SNR", "data distribution", "IID", "non-IID"],
        ["Joint user association and resource allocation for wireless hierarchical federated learning with iid and non-iid data _chunk_0"],
        topic="resource_optimization",
    ),
    case(
        "What convergence and communication guarantees are proved for local SGD on convex problems?",
        "Local SGD matches mini-batch SGD's gradient-complexity rate and achieves linear speedup in workers and mini-batch size, while reducing communication rounds by as much as a square-root factor in the total number of steps.",
        ["linear speedup", "mini-batch SGD", "communication rounds", "T"],
        ["Local SGD Converges Fast and Communicates Little_chunk_0"],
        topic="local_sgd",
    ),
    case(
        "What two kinds of heterogeneity does FedProx address, and how is it related to FedAvg?",
        "FedProx addresses statistical heterogeneity in client data and systems heterogeneity in device capabilities. It is a generalization of FedAvg that adds a proximal restriction while allowing clients to perform different amounts of work.",
        ["statistical heterogeneity", "systems heterogeneity", "proximal", "FedAvg"],
        ["MLSys-2020-federated-optimization-in-heterogeneous-networks-Paper_chunk_0"],
        topic="federated_optimization",
    ),
    case(
        "Under the PL condition, what communication behavior does the periodic-averaging local SGD analysis achieve?",
        "The method maintains an O(1/(pT)) convergence rate under the PL condition while requiring on the order of p^(1/3)T^(1/3) communication rounds, without relying on bounded gradients or strong convexity.",
        ["PL condition", "communication rounds", "p^(1/3)", "T^(1/3)"],
        ["NeurIPS-2019-local-sgd-with-periodic-averaging-tighter-analysis-and-adaptive-synchronization-Paper_chunk_0"],
        topic="local_sgd",
    ),
    case(
        "Which restrictive assumptions are removed by the Random Reshuffling analysis, and how is condition-number dependence improved?",
        "It removes the need for a small step size, bounded gradients, and a large number of epochs. For strongly convex smooth problems it improves condition-number dependence from kappa squared to kappa, with a corresponding square-root improvement in another regime.",
        ["small stepsize", "bounded gradients", "large number of epochs", "condition number"],
        ["NeurIPS-2020-random-reshuffling-simple-analysis-with-vast-improvements-Paper_chunk_0"],
        topic="shuffling_sgd",
    ),
    case(
        "What is the difference between RandomShuffle and SingleShuffle, and what assumptions are avoided by their optimal-rate analysis?",
        "RandomShuffle reshuffles examples at every epoch, whereas SingleShuffle chooses one order only at the start. The analysis covers gradient-dominated non-convex objectives without assuming each component function is convex and removes large-epoch and extra logarithmic gaps in the convex case.",
        ["RandomShuffle", "SingleShuffle", "component convexity", "large epoch"],
        ["NeurIPS-2020-sgd-with-shuffling-optimal-rates-without-component-convexity-and-large-epoch-requirements-Paper_chunk_0"],
        topic="shuffling_sgd",
    ),
    case(
        "What max-to-min stepsize ratio is necessary and sufficient for local convergence of GDA near a Stackelberg equilibrium?",
        "The local analysis shows that a ratio proportional to the local condition number kappa is necessary and sufficient, improving over the previously suggested kappa-squared ratio and narrowing the gap with practical GAN training.",
        ["stepsize ratio", "kappa", "necessary and sufficient", "Stackelberg"],
        ["On Convergence of Gradient Descent Ascent A Tight Local Analysis_chunk_0"],
        topic="minimax_optimization",
    ),
    case(
        "What convergence rate is proved for FedAvg on strongly convex smooth non-IID problems, and what slows it down?",
        "The analysis establishes an O(1/T) rate. It exposes a trade-off between communication efficiency and convergence, and shows that greater data heterogeneity slows convergence, while partial device participation can still be supported.",
        ["O(1/T)", "communication", "heterogeneity", "partial participation"],
        ["ON THE CONVERGENCE OF FEDAVG ON NON-IID_chunk_0"],
        topic="federated_optimization",
    ),
    case(
        "Why can periodic model averaging use less communication than parallel mini-batch SGD?",
        "Workers perform several local updates before their models are averaged, instead of communicating every gradient step. With a controlled averaging interval, this cuts communication while retaining convergence and training speed comparable to parallel mini-batch SGD.",
        ["model averaging", "local updates", "averaging interval", "communication"],
        ["Parallel restarted SGD with faster convergence and less communication Demystifying why model averaging works for deep learning_chunk_0"],
        topic="local_sgd",
    ),
    case(
        "What does SHARE shape at the edge, and why does that reduce cloud communication?",
        "SHARE selects edge aggregators and node assignments so each edge sees a more balanced data distribution. Effective edge aggregation then reduces the number of expensive cloud aggregations needed to reach the target accuracy.",
        ["edge aggregators", "balanced", "cloud aggregation", "communication cost"],
        ["SHARE Shaping Data Distribution at Edge for Communication-Efficient Hierarchical Federated Learning_chunk_0"],
        topic="hierarchical_fl",
    ),
    case(
        "What is a superclient in the sequential FedSeq framework, and what benefit is reported?",
        "A superclient is a subgroup of heterogeneous clients trained sequentially to emulate more centralized data exposure without sharing raw data. With a fixed communication budget, the method matches or outperforms competing federated algorithms and can improve them when combined.",
        ["superclients", "sequential training", "communication rounds", "CIFAR"],
        ["Speeding up heterogeneous federated learning with sequentially trained superclients_chunk_0"],
        topic="sequential_fl",
    ),
    case(
        "Which two failures of sequential federated learning under category and domain shifts motivate SPFL?",
        "Sequential client updates become sensitive to client order, and later clients can cause catastrophic forgetting of features learned from earlier clients. SPFL combines sequential updates with parallel aggregation to address both problems.",
        ["order sensitivity", "catastrophic forgetting", "category shift", "parallel aggregation"],
        ["SPFL Sequential Updates with Parallel Aggregation for Enhanced Federated Learning Under Category and Domain Shifts_chunk_0"],
        topic="sequential_fl",
    ),
    case(
        "What gap do the tighter lower bounds for shuffling SGD close?",
        "They give lower bounds for arbitrary weighted-average iterates with tight dependence on the number of components, epochs, and condition number. For Random Reshuffling they close the upper-lower gap in convex regimes, and for arbitrary permutations they match bounds achieved by GraB.",
        ["lower bounds", "weighted average", "condition number", "GraB"],
        ["Tighter lower bounds for shuffling sgd_chunk_0"],
        topic="shuffling_sgd",
    ),
    case(
        "What new quantity is used to obtain tighter local-SGD theory on heterogeneous data?",
        "The analysis introduces a variance notion specific to local SGD with different client data, separates identical and heterogeneous regimes, and derives improved choices for the step size and number of local iterations.",
        ["variance", "heterogeneous", "stepsize", "local iterations"],
        ["Tighter Theory for Local SGD on Identical and Heterogeneous_chunk_0"],
        topic="local_sgd",
    ),
    case(
        "Which three principles does TornadoAggregate use to control variance in a ring architecture?",
        "It uses Ring-Aware Grouping, Small Ring, and Ring Chaining. These principles reduce the high variance of ring training, yielding up to 26.7 percent higher test accuracy and near-linear scalability in the reported experiments.",
        ["Ring-Aware Grouping", "Small Ring", "Ring Chaining", "26.7"],
        ["TornadoAggregate Accurate and Scalable Federated Learning_chunk_0"],
        topic="ring_fl",
    ),
    case(
        "What is meant by unstable convergence of gradient descent?",
        "It is the observation that gradient descent can still converge in machine-learning problems when the step size violates the usual value below 2/L, but the trajectory and loss behave unstably. The paper studies the causes and linked characteristics theoretically and experimentally.",
        ["2/L", "step size", "unstable", "converges"],
        ["Understanding the unstable convergence of gradient descent_chunk_0"],
        topic="optimization_theory",
    ),
    case(
        "On CIFAR, how did FedAvg's communication requirement compare with ordinary SGD at similar accuracy?",
        "Ordinary SGD reached about 86 percent accuracy after 197,500 minibatch updates, each requiring communication, whereas FedAvg reached about 85 percent after only 2,000 communication rounds.",
        ["86", "197,500", "85", "2,000"],
        ["Communication-Efficient Learning of Deep Networks from Decentralized Data_chunk_13"],
        difficulty="hard",
        topic="federated_optimization",
    ),
    case(
        "How did the proposed UE-to-edge association method compare with greedy and random association?",
        "Across different numbers of edge servers, the proposed optimization consistently produced the lowest maximum latency. It accounts for bandwidth and accuracy constraints, unlike greedy rate-first or random association.",
        ["lowest latency", "greedy", "random", "bandwidth"],
        ["Computation_and_Communication_Resource_Optimization_for_Efficient_Hierarchical_Federated_Learning_chunk_24"],
        difficulty="hard",
        topic="resource_optimization",
    ),
    case(
        "How do epsilon and the clipping threshold affect NbAFL in the reported experiments?",
        "Increasing epsilon relaxes privacy and lowers the training loss. The clipping threshold has opposing clipping and noise effects; among the tested values 10, 15, 20, and 25, a threshold of 20 produced the best convergence result.",
        ["epsilon", "training loss", "clipping threshold", "20"],
        ["Federated_Learning_With_Differential_Privacy_Algorithms_and_Performance_Analysis(1)_chunk_13"],
        difficulty="hard",
        topic="privacy",
    ),
    case(
        "Why can FedSeq reach a target such as 70 percent accuracy with less time and communication energy?",
        "Sequential in-cluster training reduces the number of required training epochs, and only cluster heads communicate with the cloud. This outweighs the added device-to-device latency and gives the lowest time and communication energy among the compared methods.",
        ["70", "cluster heads", "training epochs", "communication energy"],
        ["FedSeq_A_Hybrid_Federated_Learning_Framework_Based_on_Sequential_In-Cluster_Training_chunk_15"],
        difficulty="hard",
        topic="sequential_fl",
    ),
    case(
        "What experimental evidence shows that inter-group heterogeneity matters more than intra-group heterogeneity in hierarchical FL?",
        "On CIFAR-10, inter-group heterogeneity caused a 6.06 percentage-point drop versus 2.50 points for intra-group heterogeneity. On Fashion-MNIST the corresponding drops were 3.18 and 0.26 points, supporting grouping that minimizes differences between groups.",
        ["6.06", "2.50", "3.18", "0.26"],
        ["iclr_2026_chunk_11"],
        difficulty="hard",
        topic="hierarchical_fl",
    ),
    case(
        "What determines optimal device-to-base-station association under non-IID wireless HFL?",
        "It depends jointly on uplink channel SNR and a weighted data-distribution distance. A device is more useful to a base station when it covers needed classes or makes the base station's aggregate data closer to IID.",
        ["uplink channel SNR", "weighted data distribution distance", "base station", "non-IID"],
        ["Joint user association and resource allocation for wireless hierarchical federated learning with iid and non-iid data _chunk_12"],
        difficulty="hard",
        topic="resource_optimization",
    ),
    case(
        "What practical heuristic is suggested for adapting FedProx's proximal coefficient mu?",
        "Increase mu when the training loss rises, and decrease it after the loss falls for several consecutive rounds. Large mu can over-constrain local updates, while very small mu may have little effect.",
        ["increase", "decrease", "loss", "mu"],
        ["MLSys-2020-federated-optimization-in-heterogeneous-networks-Paper_chunk_17"],
        difficulty="hard",
        topic="federated_optimization",
    ),
    case(
        "In the LUPA-SGD experiment, how did the chosen local-update interval differ from the earlier bound?",
        "LUPA-SGD used a local-update interval near 91, whereas the earlier square-root prescription would give about 5. The larger interval still matched synchronous SGD's final level faster in wall-clock time and retained near-linear machine speedup.",
        ["91", "5", "wall clock", "linear speedup"],
        ["NeurIPS-2019-local-sgd-with-periodic-averaging-tighter-analysis-and-adaptive-synchronization-Paper_chunk_9"],
        difficulty="hard",
        topic="local_sgd",
    ),
    case(
        "What happened when simultaneous GDA used equal 0.001 stepsizes in the GAN experiment?",
        "With equal 0.001 stepsizes for the min and max variables, simultaneous GDA converged quickly on MNIST and CIFAR-10. Decreasing the min-player step size increased the ratio and led to slower convergence and a worse final value.",
        ["0.001", "simultaneous GDA", "MNIST", "slower convergence"],
        ["On Convergence of Gradient Descent Ascent A Tight Local Analysis_chunk_12"],
        difficulty="hard",
        topic="minimax_optimization",
    ),
    case(
        "What limitations of split learning appeared as the number of IoT clients increased?",
        "Split learning often converged faster than one-epoch federated learning, but its curves were unstable, with spikes and sometimes a drop after the optimum. With 50 or 100 clients it failed to reach the centralized baseline accuracy.",
        ["unstable", "spikes", "50", "100"],
        ["Evaluation and optimization of distributed machine learning techniques for internet of things_chunk_9"],
        difficulty="hard",
        topic="systems_evaluation",
    ),
    case(
        "How do FedAvg and FedProx differ in their treatment of heterogeneous clients?",
        "FedAvg averages multiple local SGD updates and is communication efficient but can become unstable under strong statistical or systems heterogeneity. FedProx generalizes it with a proximal term and permits variable local work, producing more stable convergence in heterogeneous networks.",
        ["FedAvg", "FedProx", "proximal", "heterogeneity"],
        [
            "Communication-Efficient Learning of Deep Networks from Decentralized Data_chunk_0",
            "MLSys-2020-federated-optimization-in-heterogeneous-networks-Paper_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "How is client sequencing used differently in general sequential FL and in the hybrid FedSeq framework?",
        "General sequential FL passes a model from one client to the next and can beat parallel FL under strong heterogeneity. Hybrid FedSeq first clusters users, applies sequential training inside each cluster, and lets cluster heads communicate with the server to reduce uplink cost.",
        ["sequential FL", "clusters", "cluster heads", "heterogeneity"],
        [
            "Convergence Analysis of Sequential Federated Learning on Heterogeneous Data_chunk_0",
            "FedSeq_A_Hybrid_Federated_Learning_Framework_Based_on_Sequential_In-Cluster_Training_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "How do topology selection and edge data shaping provide complementary ways to improve hierarchical FL?",
        "Topology analysis chooses star or ring aggregation at each tier based on group size and heterogeneity. Edge data shaping instead selects aggregators and node assignments to balance group distributions, reducing costly cloud aggregation. One chooses the communication structure; the other improves the data seen within that structure.",
        ["topology", "edge data", "Star-Ring", "communication cost"],
        [
            "iclr_2026_chunk_0",
            "SHARE Shaping Data Distribution at Edge for Communication-Efficient Hierarchical Federated Learning_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "What do the two local-SGD analyses say about communication savings and data heterogeneity?",
        "The first proves that local SGD can match mini-batch SGD's gradient rate while reducing communication by up to a square-root factor. The tighter theory introduces a local-SGD-specific variance measure and shows that heterogeneous client data can substantially worsen performance and changes the best step size and local-update count.",
        ["communication", "square-root", "variance", "heterogeneous"],
        [
            "Local SGD Converges Fast and Communicates Little_chunk_0",
            "Tighter Theory for Local SGD on Identical and Heterogeneous_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "How do Random Reshuffling and the optimal-rate shuffling analysis relax earlier SGD assumptions?",
        "The Random Reshuffling work removes small-step, bounded-gradient, and large-epoch requirements and improves condition-number dependence. The optimal-rate analysis covers both reshuffling every epoch and shuffling once, including gradient-dominated non-convex objectives without component convexity.",
        ["Random Reshuffling", "bounded gradients", "SingleShuffle", "component convexity"],
        [
            "NeurIPS-2020-random-reshuffling-simple-analysis-with-vast-improvements-Paper_chunk_0",
            "NeurIPS-2020-sgd-with-shuffling-optimal-rates-without-component-convexity-and-large-epoch-requirements-Paper_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "What must be optimized jointly to reduce both latency and energy in wireless hierarchical FL?",
        "Computation and communication resources must balance completion time against energy use, while device-to-edge or device-to-base-station association must account for bandwidth and channel quality. Under non-IID data, association must additionally reduce data-distribution mismatch.",
        ["latency", "energy", "association", "data distribution"],
        [
            "Computation_and_Communication_Resource_Optimization_for_Efficient_Hierarchical_Federated_Learning_chunk_0",
            "Joint user association and resource allocation for wireless hierarchical federated learning with iid and non-iid data _chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "How do sequentially trained superclients and SPFL address different weaknesses of non-IID federated learning?",
        "Superclients sequentially expose the model to several heterogeneous clients to improve convergence within a communication budget. SPFL targets category and domain shifts specifically, combining sequential updates with parallel aggregation to reduce update-order sensitivity and catastrophic forgetting.",
        ["superclients", "SPFL", "order sensitivity", "catastrophic forgetting"],
        [
            "Speeding up heterogeneous federated learning with sequentially trained superclients_chunk_0",
            "SPFL Sequential Updates with Parallel Aggregation for Enhanced Federated Learning Under Category and Domain Shifts_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "Why does ordinary federated learning not by itself provide differential privacy, and what does NbAFL add?",
        "Federated learning keeps raw records on devices, but uploaded model parameters can still leak private information. NbAFL adds calibrated noise to client parameters before aggregation and analyzes the resulting trade-off between privacy level and convergence.",
        ["model parameters", "leak", "calibrated noise", "before aggregation"],
        [
            "Communication-Efficient Learning of Deep Networks from Decentralized Data_chunk_0",
            "Federated_Learning_With_Differential_Privacy_Algorithms_and_Performance_Analysis(1)_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "How do the diurnal multi-branch method and TornadoAggregate exploit structure beyond a standard star server?",
        "The diurnal method learns specialized branches and a temporal prior for client populations that shift with time of day. TornadoAggregate uses ring-based client traversal with grouping, small rings, and ring chaining to improve scalability while controlling variance.",
        ["temporal prior", "multi-branch", "ring", "variance"],
        [
            "Diurnal or nocturnal federated learning of multi-branch networks from periodically shifting distributions_chunk_0",
            "TornadoAggregate Accurate and Scalable Federated Learning_chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
    case(
        "How should a hierarchical FL system jointly respond to inter-group data heterogeneity and wireless constraints?",
        "It should form groups that minimize differences between group distributions, choose star or ring topology according to group size and heterogeneity, shape edge assignments so aggregators see balanced data, and allocate users using both channel SNR and data-distribution distance rather than channel quality alone.",
        ["inter-group heterogeneity", "topology", "balanced data", "SNR"],
        [
            "iclr_2026_chunk_0",
            "SHARE Shaping Data Distribution at Edge for Communication-Efficient Hierarchical Federated Learning_chunk_0",
            "Joint user association and resource allocation for wireless hierarchical federated learning with iid and non-iid data _chunk_0",
        ],
        case_type="multi_source",
        difficulty="hard",
        topic="cross_paper_comparison",
    ),
]


def normalize_excerpt(text: str, limit: int = 700) -> str:
    abstract = re.search(r"\babstract\b", text, flags=re.IGNORECASE)
    if abstract and abstract.start() < 1200:
        text = text[abstract.start():]
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def load_chunks(database: Path) -> dict[str, dict]:
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT c.chunk_id, c.content, c.page, c.chunk_index,
                   d.id AS document_id, d.source_name, d.paper_title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            """
        ).fetchall()
    finally:
        connection.close()
    return {str(row["chunk_id"]): dict(row) for row in rows}


def build(database: Path, output: Path) -> None:
    chunks = load_chunks(database)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index, spec in enumerate(CASES, start=1):
        evidence = []
        for chunk_id in spec["gold_chunk_ids"]:
            if chunk_id not in chunks:
                raise RuntimeError(f"Gold chunk does not exist: {chunk_id}")
            row = chunks[chunk_id]
            evidence.append({
                "chunk_id": chunk_id,
                "document_id": int(row["document_id"]),
                "source": row["source_name"],
                "paper_title": row["paper_title"],
                "page": row["page"],
                "chunk_index": int(row["chunk_index"]),
                "excerpt": normalize_excerpt(row["content"]),
            })
        sources = list(dict.fromkeys(item["source"] for item in evidence))
        records.append({
            "case_id": f"reviewed_academic_{index:03d}",
            "language": "en",
            "question": spec["question"],
            "case_type": spec["case_type"],
            "topic": spec["topic"],
            "difficulty": spec["difficulty"],
            "expected_answer": spec["expected_answer"],
            "expected_keywords": spec["expected_keywords"],
            "gold_sources": sources,
            "gold_evidence": evidence,
        })
    if len(records) != 50:
        raise RuntimeError(f"Expected 50 cases, found {len(records)}")
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} reviewed cases to {output}")


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
        default=Path("eval/datasets/academic_eval_50_reviewed.jsonl"),
    )
    args = parser.parse_args()
    build(args.database, args.output)


if __name__ == "__main__":
    main()
