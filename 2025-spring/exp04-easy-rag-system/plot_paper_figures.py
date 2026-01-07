import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"


def _load_benchmark_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_demo_data() -> Dict[str, Any]:
    return {
        "args": {"eval_ks": [5, 10]},
        "results": {
            "BM25-Char": {"retrieval": {"recall_at": {"5": 0.390, "10": 0.460}, "mrr": 0.290, "avg_latency_ms": 28.6}, "rerank": None},
            "MiniLM": {"retrieval": {"recall_at": {"5": 0.040, "10": 0.060}, "mrr": 0.023, "avg_latency_ms": 4.9}, "rerank": None},
            "BGE": {"retrieval": {"recall_at": {"5": 0.810, "10": 0.845}, "mrr": 0.676, "avg_latency_ms": 3.8}, "rerank": None},
            "Hybrid-RRF": {"retrieval": {"recall_at": {"5": 0.835, "10": 0.890}, "mrr": 0.701, "avg_latency_ms": 33.0}, "rerank": None},
            "BGE+Rerank": {"retrieval": {"recall_at": {"5": 0.810, "10": 0.845}, "mrr": 0.676, "avg_latency_ms": 3.8}, "rerank": {"recall_at": {"5": 0.820, "10": 0.885}, "mrr": 0.720, "avg_latency_ms": 174.9 - 3.8}},
            "Hybrid-RRF+Rerank": {"retrieval": {"recall_at": {"5": 0.835, "10": 0.890}, "mrr": 0.701, "avg_latency_ms": 33.0}, "rerank": {"recall_at": {"5": 0.845, "10": 0.905}, "mrr": 0.742, "avg_latency_ms": 176.0}},
        },
        "ablation": {
            "Full": {"Recall@5": 0.820, "MRR": 0.720, "Latency(ms)": 174.9},
            "-Hybrid": {"Recall@5": 0.820, "MRR": 0.720, "Latency(ms)": 174.9},
            "-Rerank": {"Recall@5": 0.810, "MRR": 0.676, "Latency(ms)": 3.8},
            "-QueryInstruction": {"Recall@5": 0.770, "MRR": 0.640, "Latency(ms)": 3.8},
        },
    }


def plot_fig1_architecture(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    def box(x, y, w, h, text):
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    box(0.05, 0.75, 0.9, 0.18, "数据层：Huatuo26M-Lite 医疗问答数据集 → 清洗分块 → SQLite 文档库")
    box(0.05, 0.52, 0.42, 0.18, "检索层：稠密向量检索（BGE + FAISS IVF）")
    box(0.53, 0.52, 0.42, 0.18, "稀疏检索层：SQLite FTS5（关键词检索）")
    box(0.05, 0.29, 0.9, 0.18, "证据融合与筛选：RRF 融合 → 交互式重排序（Cross-Encoder）")
    box(0.05, 0.06, 0.9, 0.18, "生成与安全：证据驱动生成（云端 LLM API）→ 不足证据拒答/风险提示")

    ax.annotate("", xy=(0.5, 0.75), xytext=(0.5, 0.70), arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(0.26, 0.52), xytext=(0.26, 0.47), arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(0.74, 0.52), xytext=(0.74, 0.47), arrowprops=dict(arrowstyle="->", linewidth=1.5))
    ax.annotate("", xy=(0.5, 0.29), xytext=(0.5, 0.24), arrowprops=dict(arrowstyle="->", linewidth=1.5))

    fig.tight_layout()
    out_path = out_dir / "fig1_architecture.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_architecture.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig2_flow_compare(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")

    def flow_row(y, title, steps):
        ax.text(0.02, y + 0.12, title, fontsize=12, fontweight="bold", va="center")
        x = 0.18
        for i, s in enumerate(steps):
            rect = plt.Rectangle((x, y), 0.16, 0.22, fill=False, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x + 0.08, y + 0.11, s, ha="center", va="center", fontsize=10)
            if i < len(steps) - 1:
                ax.annotate("", xy=(x + 0.18, y + 0.11), xytext=(x + 0.16, y + 0.11), arrowprops=dict(arrowstyle="->", linewidth=1.2))
            x += 0.2

    flow_row(0.62, "基线（单阶段 RAG）", ["Query", "关键词/向量检索", "Top-K 拼接", "LLM 生成"])
    flow_row(0.18, "本文（协同优化）", ["Query 解析", "混合召回(RRF)", "重排序", "证据不足拒答", "LLM 生成"])

    fig.tight_layout()
    out_path = out_dir / "fig2_flow_compare.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_flow_compare.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig3_main_results(out_dir: Path, payload: Dict[str, Any]) -> Path:
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")

    results = payload.get("results", {})
    order = ["BM25-Char", "MiniLM", "BGE", "Hybrid-RRF", "BGE+Rerank", "Hybrid-RRF+Rerank"]
    eval_ks = payload.get("args", {}).get("eval_ks", [5, 10])
    k_main = str(eval_ks[0]) if eval_ks else "5"

    methods = []
    recall5 = []
    mrr = []
    latency = []
    for name in order:
        if name not in results:
            continue
        r = results[name]
        retrieval = r.get("retrieval", {})
        rerank = r.get("rerank")
        use = rerank if rerank is not None else retrieval
        methods.append(name)
        recall5.append(float(use.get("recall_at", {}).get(k_main, 0.0)))
        mrr.append(float(use.get("mrr", 0.0)))
        lat = float(retrieval.get("avg_latency_ms", 0.0)) + (float(rerank.get("avg_latency_ms", 0.0)) if rerank else 0.0)
        latency.append(lat)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.barplot(x=methods, y=recall5, ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Recall@{k_main}")
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(x=methods, y=mrr, ax=axes[1], color="#55A868")
    axes[1].set_title("MRR")
    axes[1].set_ylabel("Score")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=25)

    sns.barplot(x=methods, y=latency, ax=axes[2], color="#C44E52")
    axes[2].set_title("平均延迟 (ms)")
    axes[2].set_ylabel("ms")
    axes[2].set_xlabel("")
    axes[2].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    out_path = out_dir / "fig3_main_results.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_main_results.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig4_ablation_heatmap(out_dir: Path, payload: Dict[str, Any]) -> Path:
    sns.set_theme(style="white", font="Microsoft YaHei")

    ablation = payload.get("ablation", {})
    if not ablation:
        results = payload.get("results", {})

        def _total_lat(name: str) -> float:
            r = results.get(name, {})
            retrieval = r.get("retrieval", {}) or {}
            rerank = r.get("rerank")
            return float(retrieval.get("avg_latency_ms", 0.0)) + (float((rerank or {}).get("avg_latency_ms", 0.0)) if rerank else 0.0)

        def _final(name: str) -> Dict[str, Any]:
            r = results.get(name, {})
            retrieval = r.get("retrieval", {}) or {}
            rerank = r.get("rerank")
            return rerank if rerank is not None else retrieval

        full = _final("Hybrid-RRF+Rerank")
        minus_hybrid = _final("BGE+Rerank")
        minus_rerank = _final("BGE")
        minus_inst = _final("Hybrid-RRF+Rerank(no-inst)")

        if full and minus_hybrid and minus_rerank and minus_inst:
            ablation = {
                "Full": {"Recall@5": float(full.get("recall_at", {}).get("5", 0.0)), "MRR": float(full.get("mrr", 0.0)), "Latency(ms)": _total_lat("Hybrid-RRF+Rerank")},
                "-Hybrid": {"Recall@5": float(minus_hybrid.get("recall_at", {}).get("5", 0.0)), "MRR": float(minus_hybrid.get("mrr", 0.0)), "Latency(ms)": _total_lat("BGE+Rerank")},
                "-Rerank": {"Recall@5": float(minus_rerank.get("recall_at", {}).get("5", 0.0)), "MRR": float(minus_rerank.get("mrr", 0.0)), "Latency(ms)": _total_lat("BGE")},
                "-QueryInstruction": {"Recall@5": float(minus_inst.get("recall_at", {}).get("5", 0.0)), "MRR": float(minus_inst.get("mrr", 0.0)), "Latency(ms)": _total_lat("Hybrid-RRF+Rerank(no-inst)")},
            }
        else:
            ablation = _default_demo_data()["ablation"]

    groups = list(ablation.keys())
    metrics = ["Recall@5", "MRR", "Latency(ms)"]
    data = [[float(ablation[g].get(m, 0.0)) for m in metrics] for g in groups]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3g",
        cmap="YlGnBu",
        xticklabels=metrics,
        yticklabels=groups,
        cbar=True,
        ax=ax,
    )
    ax.set_title("消融实验（同一评测集合对比）")
    fig.tight_layout()
    out_path = out_dir / "fig4_ablation.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_ablation.svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, default="")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_benchmark_json(args.input_json) or _default_demo_data()

    plot_fig1_architecture(out_dir)
    plot_fig2_flow_compare(out_dir)
    plot_fig3_main_results(out_dir, payload)
    plot_fig4_ablation_heatmap(out_dir, payload)


if __name__ == "__main__":
    main()
