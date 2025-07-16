import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

def _load_result(path: str) -> Tuple[Dict[int, float], Dict[int, List[float]], str, str, str]:
    data = json.load(open(path))
    metric_by_len = {int(k): v for k, v in data["metric_by_len"].items()}
    cos_by_len = {int(k): v for k, v in data.get("cos_by_len", {}).items()}
    return metric_by_len, cos_by_len, data["model"], data["target"], data["distractor"]

def _df_from_dir(result_dir: str):
    files = [
        f
        for f in os.listdir(result_dir)
        if f.endswith(".json") and not f.endswith("_debug.json")
    ]
    df_dict, cos_dict = {}, {}
    model = target = ""
    for fname in files:
        metric_by_len, cos_by_len, m, t, d = _load_result(os.path.join(result_dir, fname))
        df_dict[d] = pd.Series(metric_by_len)
        cos_dict[d] = pd.DataFrame.from_dict(cos_by_len, orient="index").sort_index()
        model, target = m, t
    df = pd.DataFrame(df_dict).sort_index()
    return df, model, target, cos_dict

def _pct_change(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.iloc[0]) / df.iloc[0] * 100.0

def _plot_metric(df: pd.DataFrame, model: str, target: str, out_dir: str):
    plt.figure()
    for col in df.columns:
        plt.plot(df.index, df[col], marker="o", label=col)
    plt.xlabel("History Length")
    plt.ylabel("Metric")
    plt.title(f"{model} | {target}", fontsize=8)
    plt.legend()
    path = os.path.join(out_dir, "metric.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("saved", path)

def _plot_pct_change(df: pd.DataFrame, model: str, target: str, out_dir: str):
    plt.figure()
    pct_df = _pct_change(df)
    for col in pct_df.columns:
        plt.plot(pct_df.index, pct_df[col], marker="o", label=col)
    plt.xlabel("History Length")
    plt.ylabel("% change vs h=0")
    plt.title(f"{model} | {target}", fontsize=8)
    plt.legend()
    path = os.path.join(out_dir, "metric_pct_change.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("saved", path)

def _plot_cosine(cos_dict: Dict[str, pd.DataFrame], model: str, target: str, out_dir: str):
    for distractor, cos_df in cos_dict.items():
        plt.figure()
        for h_len, row in cos_df.iterrows():
            plt.plot(cos_df.columns, row, marker="o", label=f"history {h_len}")
        plt.xlabel("Layer")
        plt.ylabel("Cosine similarity")
        plt.title(f"{model} | {target} | distractor={distractor}", fontsize=8)
        plt.legend()
        fname = f"cosine_{distractor.replace('/','_')}.png"
        path = os.path.join(out_dir, fname)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print("saved", path)

def _verify(debug_path: str):
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
        df, model, target, cos_dict = _df_from_dir(args.result_dir)
        os.makedirs(args.out_dir, exist_ok=True)
        _plot_metric(df, model, target, args.out_dir)
        _plot_pct_change(df, model, target, args.out_dir)
        _plot_cosine(cos_dict, model, target, args.out_dir)

    elif args.cmd == "verify":
        _verify(args.debug_file)

if __name__ == "__main__":
    main()
