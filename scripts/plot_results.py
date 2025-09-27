#!/usr/bin/env python3
"""
Script pour générer des graphiques :
1. Temps d'exécution par modèle.
2. Recall@k par modèle.
3. Comparaison des résultats d'extraction entre modèles.

Usage :
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
    """Génère un graphique du temps d'exécution par modèle."""
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
    plt.xlabel("Modèle")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.title("Temps d'exécution par modèle")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_file = Path(output_dir) / "execution_time_per_model.png"
    plt.savefig(output_file)
    print(f"Graphique sauvegardé : {output_file}")


def plot_recall_at_k(input_dir, output_dir):
    """Génère un graphique de Recall@k avec des barres et des moustaches pour tous les modèles."""
    data = []

    # Charger les résultats de chaque modèle
    for model_dir in Path(input_dir).iterdir():
        if model_dir.is_dir():
            eval_file = model_dir / "eval_results.json"
            if eval_file.exists():
                with open(eval_file, "r") as f:
                    results = json.load(f)
                    recall_at_k = results["retrieval"]["recall_at_k"]
                    for k, recall in recall_at_k.items():
                        data.append({"model": model_dir.name, "k": int(k), "recall": recall})

    # Convertir les données en DataFrame
    df = pd.DataFrame(data)

    # Calculer la moyenne et l'écart-type pour chaque k
    grouped = df.groupby("k")["recall"]
    mean_recall = grouped.mean()
    std_recall = grouped.std()

    # Tracer le graphique
    x = np.arange(len(mean_recall.index))  # Indices des valeurs de k

    plt.figure(figsize=(10, 6))
    plt.bar(x, mean_recall, yerr=std_recall, capsize=5, color="skyblue", alpha=0.8, label="Recall@k")

    # Ajouter les labels et la mise en forme
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("Recall@k avec variance pour tous les modèles")
    plt.xticks(x, mean_recall.index)
    plt.legend()
    plt.tight_layout()

    # Sauvegarder le graphique
    output_file = Path(output_dir) / "recall_at_k_with_variance_all_models.png"
    plt.savefig(output_file)
    print(f"Graphique sauvegardé : {output_file}")


def plot_extraction_comparison(input_dir, output_dir):
    """Génère des graphiques séparés pour les métriques d'extraction compatibles entre elles."""
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

    # Lire les résultats d'extraction pour chaque modèle
    for model_dir in Path(input_dir).iterdir():
        if model_dir.is_dir():
            eval_file = model_dir / "eval_results.json"
            if eval_file.exists():
                with open(eval_file, "r") as f:
                    results = json.load(f)
                    extraction = results.get("extraction", {})
                    # Collecter les métriques binaires
                    data_binary[model_dir.name] = [extraction.get(metric, 0) * 100 for metric in binary_metrics]
                    # Collecter les scores fuzzy
                    data_fuzzy[model_dir.name] = [extraction.get(metric, 0) for metric in fuzzy_metrics]

    # Préparer les données pour les graphiques
    models_binary = list(data_binary.keys())
    values_binary = np.array(list(data_binary.values()))

    models_fuzzy = list(data_fuzzy.keys())
    values_fuzzy = np.array(list(data_fuzzy.values()))

    # Graphique pour les métriques binaires
    x_binary = np.arange(len(binary_metrics))  # Indices des métriques binaires
    width = 0.2  # Largeur des barres

    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models_binary):
        plt.bar(x_binary + i * width, values_binary[i], width, label=model)

    plt.xlabel("Métriques binaires")
    plt.ylabel("Pourcentage (%)")
    plt.title("Comparaison des métriques binaires entre modèles")
    plt.xticks(x_binary + width * (len(models_binary) - 1) / 2, binary_metrics, rotation=45, ha="right")
    plt.legend(title="Modèles")
    plt.tight_layout()

    output_file_binary = Path(output_dir) / "extraction_comparison_binary.png"
    plt.savefig(output_file_binary)
    print(f"Graphique sauvegardé : {output_file_binary}")

    # Graphique pour les scores fuzzy
    x_fuzzy = np.arange(len(fuzzy_metrics))  # Indices des métriques fuzzy

    plt.figure(figsize=(12, 6))
    for i, model in enumerate(models_fuzzy):
        plt.bar(x_fuzzy + i * width, values_fuzzy[i], width, label=model)

    plt.xlabel("Métriques fuzzy")
    plt.ylabel("Scores")
    plt.title("Comparaison des scores fuzzy entre modèles")
    plt.xticks(x_fuzzy + width * (len(models_fuzzy) - 1) / 2, fuzzy_metrics, rotation=45, ha="right")
    plt.legend(title="Modèles")
    plt.tight_layout()

    output_file_fuzzy = Path(output_dir) / "extraction_comparison_fuzzy.png"
    plt.savefig(output_file_fuzzy)
    print(f"Graphique sauvegardé : {output_file_fuzzy}")


def main():
    parser = argparse.ArgumentParser(description="Générer des graphiques pour les résultats expérimentaux.")
    parser.add_argument("--input-dir", required=True, help="Répertoire contenant les résultats.")
    parser.add_argument("--output-dir", required=True, help="Répertoire pour sauvegarder les graphiques.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_execution_time(input_dir, output_dir)
    plot_recall_at_k(input_dir, output_dir)
    plot_extraction_comparison(input_dir, output_dir)

if __name__ == "__main__":
    main()