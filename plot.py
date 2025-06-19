import argparse
import os
import matplotlib.pyplot as plt
from pandas_helpers import df_from_dir, df_metric_pct_change

def plot_metric(df, model, target, out_dir):
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
    print("metric plot saved to", path)

def plot_pct_change(df, model, target, out_dir):
    pct_df = df_metric_pct_change(df)
    plt.figure()
    
    for col in pct_df.columns:
        plt.plot(pct_df.index, pct_df[col], marker="o", label=col)
    
    plt.xlabel("History Length")
    plt.ylabel("Percentage Change")
    plt.title(f"{model} | {target}", fontsize=8)
    plt.legend()
    
    path = os.path.join(out_dir, "metric_pct_change.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print("pct change plot saved to", path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--out_dir", default="plots")
    args = ap.parse_args()
    
    df, model, target = df_from_dir(args.result_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    
    plot_metric(df, model, target, args.out_dir)
    plot_pct_change(df, model, target, args.out_dir)

if __name__ == "__main__":
    main()