#!/usr/bin/env python3
"""
Évaluer la performance de FAISS (retrieval) et du LLM (extraction).

Usage:
  python3 scripts/evaluate_faiss_llm.py --input search_result.csv --out-json eval_results.json

Le fichier CSV attendu contient au minimum les colonnes:
  label, score, city, city_orig, date, date_orig, venue, venue_orig,
  address, address_orig, capacity, capacity_orig

Le script calcule Recall@k (pour k=1,3,5,10 par défaut), MRR, et pour
les prédictions top-1 il calcule l'exact match pour date/capacity et la
similarité fuzzy pour texte (venue, address) en utilisant RapidFuzz.
"""

import argparse
import json
from collections import defaultdict
from typing import Dict, List

import pandas as pd
from rapidfuzz import fuzz

def compute_retrieval_metrics(df: pd.DataFrame, group_col: str = "label",
                              pred_col: str = "city", truth_col: str = "label",
                              ks=(1, 3)) -> Dict:
    groups = df.groupby(group_col)
    n_queries = len(groups)
    recall_at_k = {k: 0 for k in ks}
    rr_sum = 0.0

    for _, group in groups:
        group = group.sort_values("score", ascending=False)
        is_correct = group[pred_col] == group[truth_col]

        ranks = is_correct[is_correct].index
        if not ranks.empty:
            rr_sum += 1 / (ranks[0] + 1)

        for k in ks:
            if k <= len(group):
                recall_at_k[k] += is_correct.iloc[:k].any()

    recall_at_k = {k: v / n_queries for k, v in recall_at_k.items()}
    mrr = rr_sum / n_queries
    return {"recall_at_k": recall_at_k, "mrr": mrr}

def compute_extraction_metrics(df: pd.DataFrame, fuzzy_threshold: int = 80) -> Dict:
    stats = defaultdict(int)
    fuzzy_scores = defaultdict(list)

    for _, row in df.iterrows():
        # Exact match for date
        stats["date_exact"] += row["date"] == row["date_orig"]

        # Exact match for capacity
        try:
            stats["capacity_exact"] += int(row["capacity"]) == int(row["capacity_orig"])
        except (ValueError, TypeError):
            pass

        # Fuzzy match for venue
        venue_score = fuzz.token_sort_ratio(str(row["venue"]), str(row["venue_orig"]))
        fuzzy_scores["venue"].append(venue_score)
        stats["venue_above_threshold"] += venue_score >= fuzzy_threshold

        # Fuzzy match for address
        address_score = fuzz.token_sort_ratio(str(row["address"]), str(row["address_orig"]))
        fuzzy_scores["address"].append(address_score)
        stats["address_above_threshold"] += address_score >= fuzzy_threshold

    n = len(df)
    return {
        "date_exact": stats["date_exact"] / n,
        "capacity_exact": stats["capacity_exact"] / n,
        "venue_above_threshold": stats["venue_above_threshold"] / n,
        "address_above_threshold": stats["address_above_threshold"] / n,
        "venue_mean_fuzzy": sum(fuzzy_scores["venue"]) / n,
        "address_mean_fuzzy": sum(fuzzy_scores["address"]) / n,
    }

def main():
    parser = argparse.ArgumentParser(description="Évaluer FAISS et LLM à partir d'un fichier CSV.")
    parser.add_argument("--input", required=True, help="Chemin vers le fichier CSV d'entrée.")
    parser.add_argument("--out-json", help="Chemin pour sauvegarder les résultats en JSON.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Inclure Recall@2 dans les métriques
    retrieval_metrics = compute_retrieval_metrics(df, ks=(1, 2, 3))
    extraction_metrics = compute_extraction_metrics(df)

    results = {
        "retrieval": retrieval_metrics,
        "extraction": extraction_metrics,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()