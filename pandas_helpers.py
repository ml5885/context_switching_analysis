import os
import json
import pandas as pd

def load_result(path):
    data = json.load(open(path))
    metric_by_len = {int(k): v for k, v in data["metric_by_len"].items()}
    return metric_by_len, data["model"], data["target"], data["distractor"]

def df_from_dir(result_dir):
    files = [
        f for f in os.listdir(result_dir)
        if f.endswith(".json") and not f.endswith("_debug.json")
    ]
    df_dict = {}
    model = target = None
    for fname in files:
        metric_by_len, m, t, d = load_result(os.path.join(result_dir, fname))
        df_dict[d] = pd.Series(metric_by_len)
        model = m
        target = t
    df = pd.DataFrame(df_dict).sort_index()
    return df, model, target

def df_metric_pct_change(df):
    return (df - df.iloc[0]) / df.iloc[0] * 100
