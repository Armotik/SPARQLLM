#!/usr/bin/env python3
"""
Évaluer la performance de FAISS (retrieval) et du LLM (extraction) + hit@k spécifique.

Usage:
    python3 scripts/evaluate_faiss_llm.py --input search_result.csv --out-json eval_results.json [--hit-ks 1 2 3]

Colonnes attendues (certaines optionnelles):
    label, score, city, city_orig, date, date_orig, venue, venue_orig,
    address, address_orig, capacity, capacity_orig, type_orig

Métriques calculées:
    - recall_at_k & mrr (retrieval basés sur city vs label)
    - hit@k : succès si (city_orig == label ET type_orig == 'cultural') dans le top-k (success@k conditionnel)
    - extraction: exact match date/capacity + fuzzy similarity (RapidFuzz) venue/address

Notes:
    - hit@k est ignoré si les colonnes nécessaires (city_orig, type_orig) manquent.
    - recall@k ici correspond à un success@k (1 si un pertinent dans top-k) vu hypothèse 1 vérité / label.
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
    """Compute success-like recall@k (1 if any pred matches truth in top-k) + MRR."""
    groups = df.groupby(group_col)
    n_queries = len(groups)
    recall_at_k = {k: 0 for k in ks}
    rr_sum = 0.0

    for _, group in groups:
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        is_correct = group[pred_col] == group[truth_col]

        # Reciprocal rank of first correct
        correct_positions = is_correct[is_correct].index.tolist()
        if correct_positions:
            rr_sum += 1 / (correct_positions[0] + 1)

        for k in ks:
            head = is_correct.iloc[:k]
            recall_at_k[k] += bool(head.any())

    recall_at_k = {k: v / n_queries for k, v in recall_at_k.items()}
    mrr = rr_sum / n_queries if n_queries else 0.0
    return {"recall_at_k": recall_at_k, "mrr": mrr}


def compute_hit_at_k(df: pd.DataFrame, group_col: str = "label", ks=(1, 2, 3),
                     city_orig_col: str = "city_orig", type_col: str = "type_orig",
                     required_type: str = "cultural") -> Dict:
    """Compute hit@k per query (city) where a hit is city_orig == label AND type_orig == required_type.
    Returns dict hit_at_k {k: ratio}.
    If needed columns missing, returns empty dict.
    """
    missing = [c for c in [group_col, city_orig_col, type_col, 'score'] if c not in df.columns]
    if missing:
        return {"hit_at_k": {}, "hit_definition": f"Skipped (missing columns: {missing})"}

    groups = df.groupby(group_col)
    n_queries = len(groups)
    hit_counts = {k: 0 for k in ks}

    for label, group in groups:
        group = group.sort_values("score", ascending=False).reset_index(drop=True)
        cond = (group[city_orig_col] == label) & (group[type_col].str.lower() == required_type.lower())
        for k in ks:
            if bool(cond.iloc[:k].any()):
                hit_counts[k] += 1

    hit_at_k = {k: hit_counts[k] / n_queries for k in ks}
    return {"hit_at_k": hit_at_k, "hit_definition": f"city_orig == label AND {type_col} == '{required_type}'"}

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


def compute_simple_precision(df: pd.DataFrame,
                             label_col: str = "label",
                             city_col: str = "city_orig",
                             type_col: str = "type_orig",
                             required_type: str = "cultural") -> Dict:
    """Calcule une précision globale très simple:
    - row_precision: proportion de lignes où (label == city_orig) ET (type_orig == required_type)
    - city_full_precision: fraction des villes dont TOUTES les lignes sont correctes
    - city_at_least1: fraction des villes avec ≥ 1 ligne correcte
    - row_precision_pct: pourcentage (100*row_precision)

    Hypothèses: le DataFrame contient au moins label_col / city_col / type_col.
    Retourne des 0.0 si DataFrame vide ou colonnes absentes.
    """
    needed = [label_col, city_col, type_col]
    if any(c not in df.columns for c in needed) or df.empty:
        return {
            "row_precision": 0.0,
            "row_precision_pct": 0.0,
            "city_full_precision": 0.0,
            "city_at_least1": 0.0,
            "definition": f"(label == city_orig AND type_orig == '{required_type}')"
        }

    correct_mask = (df[label_col] == df[city_col]) & (df[type_col].str.lower() == required_type.lower())
    row_precision = float(correct_mask.mean()) if len(correct_mask) else 0.0

    # Par ville
    full = 0
    atleast1 = 0
    groups = list(df.groupby(label_col))
    for _, g in groups:
        g_mask = (g[label_col] == g[city_col]) & (g[type_col].str.lower() == required_type.lower())
        c = int(g_mask.sum())
        if c == len(g):
            full += 1
        if c > 0:
            atleast1 += 1
    n_cities = len(groups) or 1

    return {
        "row_precision": row_precision,
        "row_precision_pct": row_precision * 100.0,
        "city_full_precision": full / n_cities,
        "city_at_least1": atleast1 / n_cities,
        "definition": f"(label == city_orig AND type_orig == '{required_type}')"
    }

def main():
    parser = argparse.ArgumentParser(description="Évaluer FAISS et LLM à partir d'un fichier CSV.")
    parser.add_argument("--input", required=True, help="Chemin vers le fichier CSV d'entrée.")
    parser.add_argument("--out-json", help="Chemin pour sauvegarder les résultats en JSON.")
    parser.add_argument("--hit-ks", nargs="*", type=int, default=[1, 2, 3], help="Valeurs k pour hit@k (par défaut: 1 2 3)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # Retrieval metrics (success-like recall@k + MRR)
    retrieval_metrics = compute_retrieval_metrics(df, ks=(1, 2, 3))

    # Hit@k metrics (conditional success) if columns present
    hit_metrics = compute_hit_at_k(df, ks=tuple(args.hit_ks))
    extraction_metrics = compute_extraction_metrics(df)
    simple_precision = compute_simple_precision(df)

    results = {
        "retrieval": retrieval_metrics,
        "hit": hit_metrics,
        "extraction": extraction_metrics,
        "simple_precision": simple_precision,
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()