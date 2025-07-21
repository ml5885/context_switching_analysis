import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

def load_result(path):
    data = json.load(open(path))
    metric_by_len = {int(k): v for k, v in data["metric_by_len"].items()}
    cos_by_len = {int(k): v for k, v in data.get("cos_by_len", {}).items()}
    return (
        metric_by_len,
        cos_by_len,
        data["model"],
        data["target"],
        data["distractor"],
        data["metric_name"],
    )

def df_from_dir(result_dir):
    paths = []
    for root, _, files in os.walk(result_dir):
        for f in files:
            if f.endswith(".json") and not f.endswith("_debug.json"):
                paths.append(os.path.join(root, f))

    df_dict, cos_dict = {}, {}
    model = target = metric = ""
    for path in paths:
        metric_by_len, cos_by_len, m, t, d, met = load_result(path)
        df_dict[d] = pd.Series(metric_by_len)
        cos_dict[d] = pd.DataFrame.from_dict(cos_by_len, orient="index").sort_index()
        model, target, metric = m, t, met

    df = pd.DataFrame(df_dict).sort_index()
    return df, model, target, metric, cos_dict

def pct_change(df):
    return (df - df.iloc[0]) / df.iloc[0] * 100.0

def plot(series_dict, xlabel, ylabel, title, out_dir, fname, legend_title):
    plt.figure(figsize=(8, 6))

    for label, series in series_dict.items():
        plt.plot(series.index, series.values, marker="o", label=label)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=10)

    plt.legend(
        title=legend_title,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.4),
        fontsize=10,
        title_fontsize=11,
        ncol=3,
        frameon=True
    )

    ticks = next(iter(series_dict.values())).index
    plt.xticks(ticks, fontsize=10)
    plt.tight_layout(rect=[0, 0.12, 1, 1])

    path = os.path.join(out_dir, fname)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    print("saved", path)

def plot_metric(df, model, target, metric, out_dir):
    title = f"{model} | target={target} | metric={metric}"

    plot(
        df.to_dict("series"),
        "History Length",
        metric,
        title,
        out_dir,
        "metric.png",
        legend_title="distractor"
    )

def plot_pct_change(df, model, target, metric, out_dir):
    pct = pct_change(df)
    title = f"{model} | target={target} | % change in {metric}"
    ylabel = f"% change vs h=0 ({metric})"

    plot(
        pct.to_dict("series"),
        "History Length",
        ylabel,
        title,
        out_dir,
        "metric_pct_change.png",
        legend_title="distractor"
    )

def plot_cosine(cos_dict, model, target, metric, out_dir):
    for distractor, cos_df in cos_dict.items():
        title = f"{model} | target={target} | distractor={distractor} | metric={metric}"
        fname = f"cosine_{distractor.replace('/', '_')}.png"

        fig, ax = plt.subplots(figsize=(10, 6))
        history_lens = list(cos_df.index)

        cmap = plt.get_cmap("viridis")
        norm = mpl.colors.Normalize(vmin=min(history_lens), vmax=max(history_lens))
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)

        for hist_len in history_lens:
            color = cmap(norm(hist_len))
            ax.plot(cos_df.columns, cos_df.loc[hist_len], marker="o", color=color, label=f"History {hist_len}")

        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Cosine similarity", fontsize=12)
        ax.set_title(title, fontsize=10)

        cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, aspect=40)
        cbar.set_label("History Length", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xticks(cos_df.columns)
        ax.tick_params(axis='x', labelsize=10)

        fig.tight_layout(rect=[0, 0, 1, 1])

        path = os.path.join(out_dir, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        print("saved", path)

def verify(debug_path):
    with open(debug_path) as f:
        data = json.load(f)

    res = defaultdict(lambda: {"correct": 0, "total": 0})
    for ex in data.get("debug_examples", []):
        cfg = ex["config"]
        key = (cfg["history_len"], cfg["distractor_dataset"])
        res[key]["total"] += 1
        if ex["model_prediction"] == ex["expected_answer"]:
            res[key]["correct"] += 1

    header = f"Recomputed accuracy → {debug_path}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'hist_len':<8} {'distractor':<20} {'acc':>6}")
    for (h_len, dis), vals in sorted(res.items()):
        acc = vals["correct"] / vals["total"]
        print(f"{h_len:<8} {dis:<20} {acc:>6.4f}")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_metric = sub.add_parser("metric", help="plots from result_dir")
    p_metric.add_argument("--result_dir", required=True)
    p_metric.add_argument("--out_dir", default="plots")

    p_verify = sub.add_parser("verify", help="recompute accuracy from *_debug.json")
    p_verify.add_argument("--debug_file", required=True)

    args = ap.parse_args()

    if args.cmd == "metric":
        df, model, target, metric, cos_dict = df_from_dir(args.result_dir)
        os.makedirs(args.out_dir, exist_ok=True)
        plot_metric(df, model, target, metric, args.out_dir)
        plot_pct_change(df, model, target, metric, args.out_dir)
        plot_cosine(cos_dict, model, target, metric, args.out_dir)

    elif args.cmd == "verify":
        verify(args.debug_file)

if __name__ == "__main__":
    main()
    
# Example usage:
# python viz.py metric --result_dir test_results/meta-llama@Llama-2-7b-chat-hf__mmlu --out_dir plots_mmlu
# python viz.py metric --result_dir test_results/meta-llama@Llama-2-7b-chat-hf__rotten_tomatoes --out_dir plots_rotten_tomatoes
# python viz.py metric --result_dir test_results/meta-llama@Llama-2-7b-chat-hf__tweetqa --out_dir plots_tweetqa