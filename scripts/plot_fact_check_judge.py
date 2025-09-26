#!/usr/bin/env python3
"""Plot results produced by experiment_fact_check_judge.py

Basic usage:
    ./scripts/plot_fact_check_judge.py \
         --summary out/experiments-fcj/summary.csv \
         --out-dir out/experiments-fcj/plots

Outputs:
    - wall_time_vs_rows.png : total wall time vs number of facts (lines per n_models)
    - time_per_row.png      : average time per fact (grouped bars)
    - decisions_stack.png   : distribution of confirmed/refuted/tie per combination
    - heatmap_time_per_row_[n_models].png : heatmaps per n_models (dbpedia_limit x search_limit)

"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ajouts dérivés
    df["time_per_row"] = df.apply(lambda r: (r.wall_time_sec / r.total_rows) if r.total_rows else 0.0, axis=1)
    df["combo"] = (
        "db" + df.dbpedia_limit.astype(str)
        + "_se" + df.search_limit.astype(str)
        + "_nm" + df.n_models.astype(str)
    )
    return df


def plot_wall_time(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for n_models, g in df.groupby("n_models"):
        g_sorted = g.sort_values("total_rows")
        ax.plot(
            g_sorted.total_rows,
            g_sorted.wall_time_sec,
            marker="o",
            label=f"n_models={n_models}"
        )
    ax.set_xlabel("Number of facts (total_rows)")
    ax.set_ylabel("Total time (s)")
    ax.set_title("Total time vs number of facts")
    ax.grid(alpha=0.3)
    ax.legend()
    out = out_dir / "wall_time_vs_rows.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[write] {out}")
    plt.close(fig)


def plot_time_per_row(df: pd.DataFrame, out_dir: Path) -> None:
    # Regrouper par (dbpedia_limit, search_limit, n_models)
    df2 = df.copy()
    df2["group_label"] = (
        "db=" + df2.dbpedia_limit.astype(str)
        + " se=" + df2.search_limit.astype(str)
    )
    pivot = df2.pivot_table(
        index="group_label",
        columns="n_models",
        values="time_per_row",
        aggfunc="mean"
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for i, col in enumerate(pivot.columns):
        ax.bar(
            x + i * width,
            pivot[col].values,
            width=width,
            label=f"n_models={col}"
        )
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel("Time / fact (s)")
    ax.set_title("Average time per fact")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    out = out_dir / "time_per_row.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[write] {out}")
    plt.close(fig)


def plot_decisions(df: pd.DataFrame, out_dir: Path) -> None:
    # Barres empilées par combinaison (db,se) pour n_models distincts
    df3 = df.copy()
    df3["base_combo"] = (
        "db=" + df3.dbpedia_limit.astype(str)
        + " se=" + df3.search_limit.astype(str)
        + " nm=" + df3.n_models.astype(str)
    )
    df3 = df3.sort_values(["dbpedia_limit", "search_limit", "n_models"])
    labels = df3.base_combo.tolist()
    confirmed = df3.confirmed_rows.to_numpy()
    refuted = df3.refuted_rows.to_numpy()
    tie = df3.tie_rows.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, confirmed, label="confirmed", color="#2c7fb8")
    ax.bar(x, refuted, bottom=confirmed, label="refuted", color="#d95f0e")
    ax.bar(x, tie, bottom=confirmed + refuted, label="tie", color="#969696")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Number of decisions")
    ax.set_title("Decision distribution per combination")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = out_dir / "decisions_stack.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"[write] {out}")
    plt.close(fig)


def plot_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    # Heatmap time_per_row pour chaque n_models
    for n_models, g in df.groupby("n_models"):
        pivot = g.pivot_table(
            index="dbpedia_limit",
            columns="search_limit",
            values="time_per_row",
            aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("search_limit")
        ax.set_ylabel("dbpedia_limit")
        ax.set_title(f"Time per row (s) heatmap n_models={n_models}")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if val > pivot.values.mean() else "black",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, shrink=0.75, label="s/row")
        out = out_dir / f"heatmap_time_per_row_nm{n_models}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=140)
        print(f"[write] {out}")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot fact-check judge experiments summary")
    ap.add_argument("--summary", default="out/experiments-fcj/summary.csv")
    ap.add_argument("--out-dir", default="out/experiments-fcj/plots")
    ap.add_argument("--no-heatmap", action="store_true", help="Disable heatmaps")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Summary CSV not found: {summary_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(summary_path)
    if df.empty:
        raise SystemExit("Empty summary – nothing to plot.")

    plot_wall_time(df, out_dir)
    plot_time_per_row(df, out_dir)
    plot_decisions(df, out_dir)
    if not args.no_heatmap:
        plot_heatmaps(df, out_dir)
    print("[done] Plots generated.")


if __name__ == "__main__":
    main()
