import argparse
import json
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
from tqdm import tqdm

def load_result(path):
    with open(path) as f:
        data = json.load(f)
    
    return {
        "metric_by_len": {int(k): v for k, v in data["metric_by_len"].items()},
        "cos_by_len": {int(k): [np.array(x) for x in v] for k, v in data["cos_by_len"].items()},
        "metric_name": data["metric_name"],
        "model": data["model"],
        "target": data["target"],
        "distractor": data["distractor"]
    }

def plot_metric(data, out_dir):
    xs = sorted(data["metric_by_len"])
    ys = [np.mean(data["metric_by_len"][x]) for x in xs]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("History Length")
    plt.ylabel(data["metric_name"].capitalize())

    title = f"{data['model']} | {data['target']} vs {data['distractor']}"
    plt.title(title, fontsize=8)

    fname = os.path.join(out_dir, "metric.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("metric plot saved to", fname)

def plot_cos(data, out_dir):
    plt.figure()
    for h, sims in data["cos_by_len"].items():
        plt.plot(np.mean(sims, axis=0), label=str(h))

    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.legend()

    title = f"{data['model']} | {data['target']} vs {data['distractor']}"
    plt.title(title, fontsize=8)

    fname = os.path.join(out_dir, "cosine.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print("cosine plot saved to", fname)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--result",
        required=True,
        help="path to result json or dir containing them"
    )
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()

    paths = (
        [args.result]
        if os.path.isfile(args.result)
        else glob.glob(os.path.join(args.result, "*.json"))
    )

    # Filter out debug files
    paths = [p for p in paths if not p.endswith("_debug.json")]

    os.makedirs(args.out_dir, exist_ok=True)

    for p in tqdm(paths, desc="Generating plots"):
        data = load_result(p)
        base = os.path.splitext(os.path.basename(p))[0]
        out_sub = os.path.join(args.out_dir, base)
        os.makedirs(out_sub, exist_ok=True)
        plot_metric(data, out_sub)
        plot_cos(data, out_sub)
    
    print(f"\nDone. All plots are in {args.out_dir}/")

if __name__ == "__main__":
    main()