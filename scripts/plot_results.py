#!/usr/bin/env python3
"""
Script to generate graphs:
1. Execution time by model.
2. Recall@k by model.
3. Comparison of extraction results between models.

Usage:
  python3 scripts/plot_results.py --input-dir out/ --output-dir plots/
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_execution_time(input_dir, output_dir):
    """Generate a plot of execution time by model."""
    data = []

    for model_dir in Path(input_dir).iterdir():
        if model_dir.is_dir():
            time_file = model_dir / "execution_time.txt"
            if time_file.exists():
                with open(time_file, "r") as f:
                    line = f.readline()
                    time = float(line.split(":")[-1].strip().split()[0])
                    data.append({"model": model_dir.name, "execution_time": time})

    df = pd.DataFrame(data)
    df = df.sort_values("execution_time")

    plt.figure(figsize=(10, 6))
    plt.bar(df["model"], df["execution_time"], color="skyblue")
    plt.xlabel("Model")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time by Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_file = Path(output_dir) / "execution_time_per_model.png"
    plt.savefig(output_file)
    print(f"Plot saved: {output_file}")


def plot_recall_at_k(input_dir, output_dir):
    """Generate a Recall@k plot with error bars for all models."""
    data = []

    # Load results from each model
    for model_dir in Path(input_dir).iterdir():
        if model_dir.is_dir():
            eval_file = model_dir / "eval_results.json"
            if eval_file.exists():
                with open(eval_file, "r") as f:
                    results = json.load(f)
                    recall_at_k = results["retrieval"]["recall_at_k"]
                    for k, recall in recall_at_k.items():
                        data.append({"model": model_dir.name, "k": int(k), "recall": recall})

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Calculate mean and standard deviation for each k
    grouped = df.groupby("k")["recall"]
    mean_recall = grouped.mean()
    std_recall = grouped.std()

    # Plot the graph
    x = np.arange(len(mean_recall.index))  # Indices for k values

    plt.figure(figsize=(10, 6))
    plt.bar(x, mean_recall, yerr=std_recall, capsize=5, color="skyblue", alpha=0.8, label="Recall@k")

    # Add labels and formatting
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("Recall@k with Variance for All Models")
    plt.xticks(x, mean_recall.index)
    plt.legend()
    plt.tight_layout()

    # Save the graph
    output_file = Path(output_dir) / "recall_at_k_with_variance_all_models.png"
    plt.savefig(output_file)
    print(f"Plot saved: {output_file}")


def plot_extraction_comparison(input_dir, output_dir):
    """Generate separate plots for compatible extraction metrics."""
    binary_metrics = [
        "date_exact",
        "capacity_exact",
        "venue_above_threshold",
        "address_above_threshold"
    ]
    fuzzy_metrics = [
        "venue_mean_fuzzy",
        "address_mean_fuzzy"
    ]

    data_binary = {}
    data_fuzzy = {}

    # Read extraction results for each model
    for model_dir in Path(input_dir).iterdir():
        if model_dir.is_dir():
            eval_file = model_dir / "eval_results.json"
            if eval_file.exists():
                with open(eval_file, "r") as f:
                    results = json.load(f)
                    extraction = results.get("extraction", {})
                    # Collect binary metrics
                    data_binary[model_dir.name] = [extraction.get(metric, 0) * 100 for metric in binary_metrics]
                    # Collect fuzzy scores
                    data_fuzzy[model_dir.name] = [extraction.get(metric, 0) for metric in fuzzy_metrics]

    # Prepare data for plots
    models_binary = list(data_binary.keys())
    values_binary = np.array(list(data_binary.values()))

    models_fuzzy = list(data_fuzzy.keys())
    values_fuzzy = np.array(list(data_fuzzy.values()))

    # Plot for binary metrics
    x_binary = np.arange(len(binary_metrics))  # Indices for binary metrics
    width = 0.2  # Bar width

    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models_binary):
        plt.bar(x_binary + i * width, values_binary[i], width, label=model)

    plt.xlabel("Binary Metrics")
    plt.ylabel("Percentage (%)")
    plt.title("Comparison of Binary Metrics Across Models")
    plt.xticks(x_binary + width * (len(models_binary) - 1) / 2, binary_metrics, rotation=45, ha="right")
    plt.legend(title="Models")
    plt.tight_layout()

    output_file_binary = Path(output_dir) / "extraction_comparison_binary.png"
    plt.savefig(output_file_binary)
    print(f"Plot saved: {output_file_binary}")

    # Plot for fuzzy scores
    x_fuzzy = np.arange(len(fuzzy_metrics))  # Indices for fuzzy metrics

    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models_fuzzy):
        plt.bar(x_fuzzy + i * width, values_fuzzy[i], width, label=model)

    plt.xlabel("Fuzzy Metrics")
    plt.ylabel("Scores")
    plt.title("Comparison of Fuzzy Scores Across Models")
    plt.xticks(x_fuzzy + width * (len(models_fuzzy) - 1) / 2, fuzzy_metrics, rotation=45, ha="right")
    plt.legend(title="Models")
    plt.tight_layout()

    output_file_fuzzy = Path(output_dir) / "extraction_comparison_fuzzy.png"
    plt.savefig(output_file_fuzzy)
    print(f"Plot saved: {output_file_fuzzy}")


def plot_row_precision_per_model(input_dir, output_dir):
    """Plot row_precision_pct (simple_precision) per model as a bar chart."""
    data = []

    for model_dir in Path(input_dir).iterdir():
        if model_dir.is_dir():
            # support multiple possible eval filenames
            eval_file = None
            for name in ("eval_results.json", "eval.json", "eval.jsonl"):
                p = model_dir / name
                if p.exists():
                    eval_file = p
                    break
            if not eval_file:
                continue
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except Exception:
                continue
            sp = results.get("simple_precision") or {}
            pct = sp.get("row_precision_pct")
            if pct is None:
                # maybe older files store row_precision as fraction
                rp = sp.get("row_precision")
                if rp is not None:
                    pct = float(rp) * 100.0
            if pct is not None:
                data.append({"model": model_dir.name, "row_precision_pct": float(pct)})

    if not data:
        print("No simple_precision data found for any model; skipping precision plot.")
        return

    df = pd.DataFrame(data).sort_values("row_precision_pct", ascending=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df["model"], df["row_precision_pct"], color="seagreen")
    plt.xlabel("Model")
    plt.ylabel("Row Precision (%)")
    plt.title("Row Precision (%) per Model")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_file = Path(output_dir) / "row_precision_per_model.png"
    plt.savefig(output_file)
    print(f"Plot saved: {output_file}")


def plot_aggregate_hit_at_k(input_dir, output_dir, ks=(1, 2, 3)):
    """Aggregate hit@k across all model runs (all models/confounded).
    It looks for a search CSV in each subfolder (search.csv or search_result.csv) and
    recomputes hit@k using condition: (city_orig == label) AND (type_orig == 'cultural').
    """
    total_queries = 0
    total_hits = {k: 0 for k in ks}

    csv_names = ("search.csv", "search_result.csv", "search_results.csv")
    for model_dir in Path(input_dir).iterdir():
        if not model_dir.is_dir():
            continue
        # find csv file
        csv_file = None
        for name in csv_names:
            p = model_dir / name
            if p.exists():
                csv_file = p
                break
        # also accept files under nested folders (e.g., out/model/search.csv)
        if csv_file is None:
            # try to find any csv in the folder
            for p in model_dir.glob("*.csv"):
                csv_file = p
                break
        if csv_file is None:
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception:
            print(f"Warning: could not read {csv_file}")
            continue

        if df.empty or 'label' not in df.columns:
            continue

        groups = df.groupby('label')
        n_queries = len(groups)
        total_queries += n_queries

        for label, group in groups:
            group = group.sort_values('score', ascending=False).reset_index(drop=True)
            # safety: lower type column
            if 'type_orig' not in group.columns or 'city_orig' not in group.columns:
                continue
            cond = (group['city_orig'] == label) & (group['type_orig'].astype(str).str.lower() == 'cultural')
            for k in ks:
                if bool(cond.iloc[:k].any()):
                    total_hits[k] += 1

    if total_queries == 0:
        print("No queries found across model outputs; skipping aggregate hit@k plot.")
        return

    hit_at_k = {k: total_hits[k] / total_queries for k in ks}

    # Plot
    labels = [str(k) for k in ks]
    values = [hit_at_k[k] * 100.0 for k in ks]

    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, values, color='coral')
    for i, v in enumerate(values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.xticks(x, labels)
    plt.ylim(0, 100)
    plt.xlabel('k')
    plt.ylabel('Hit@k (%) — aggregated across models')
    plt.title('Aggregate Hit@k (city_orig==label AND type_orig==cultural)')
    plt.tight_layout()

    output_file = Path(output_dir) / 'aggregate_hit_at_k.png'
    plt.savefig(output_file)
    print(f"Plot saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for experimental results.")
    parser.add_argument("--input-dir", required=True, help="Directory containing the results.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the plots.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_execution_time(input_dir, output_dir)
    plot_recall_at_k(input_dir, output_dir)
    plot_extraction_comparison(input_dir, output_dir)
    # plot_row_precision_per_model(input_dir, output_dir)
    plot_aggregate_hit_at_k(input_dir, output_dir)

if __name__ == "__main__":
    main()