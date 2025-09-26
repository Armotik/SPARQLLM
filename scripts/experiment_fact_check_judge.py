#!/usr/bin/env python3
"""
Automatise des expériences sur la requête fact-check-judge.sparql en faisant varier:
- le LIMIT DBpedia (nombre de (pays, capitale))
- le limit du search DuckDuckGo
- le nombre de modèles (LLM-as-Judge) utilisés (prend les N premiers d'une liste base)

Mesures collectées par run:
- wall_time_sec (temps d'exécution slm-run)
- total_rows (nombre de faits jugés = lignes)
- confirmed_rows / refuted_rows / tie_rows (selon majorityDecision)

Sorties:
- Un CSV de résultats par run dans out/experiments-fcj/results/
- Un CSV récapitulatif dans out/experiments-fcj/summary.csv

Exemple:
  ./scripts/experiment_fact_check_judge.py \
    --config config.ini \
    --query queries/web/fact-check-judge.sparql \
    --dbpedia-limits 3,5,10 \
    --search-limits 1,3 \
    --n-models 1,3 \
    --models "llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-20b" \
    --debug
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def find_slm_run(explicit: str | None) -> str:
    if explicit:
        return explicit
    venv_bin = Path("./venv/bin/slm-run").resolve()
    if venv_bin.exists():
        return str(venv_bin)
    return "slm-run"  # fallback to PATH


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def set_dbpedia_limit(query: str, limit: int) -> str:
    # Remplace la limite de la sous-requête DBpedia: `} LIMIT <num>`
    return re.sub(r"\} LIMIT \d+", f"}} LIMIT {limit}", query, count=1)


def set_search_limit(query: str, limit: int) -> str:
    # Remplace le paramètre "limit" de DuckDuckGo: '"limit", <num>' (première occurrence)
    return re.sub(r"(\"limit\",\s*)\d+", rf"\g<1>{limit}", query, count=1)


def set_models(query: str, models: List[str]) -> str:
    # Remplace le bloc VALUES ?groqModel { ... }
    pattern = re.compile(r"(VALUES\s+\?groqModel\s*\{)([\s\S]*?)(\})", re.MULTILINE)

    def repl(m: re.Match[str]) -> str:
        prefix, _, suffix = m.group(1), m.group(2), m.group(3)
        body = "\n".join([f"    \"{model}\"" for model in models])
        return f"{prefix}\n{body}\n  {suffix}"

    new_query, n = pattern.subn(repl, query, count=1)
    if n == 0:
        raise RuntimeError("Bloc VALUES ?groqModel { … } introuvable dans la requête")
    return new_query


def run_slm(slm_bin: str, config: str, query_file: Path, output_csv: Path, debug: bool = False) -> float:
    cmd = [slm_bin, "--config", config, "-f", str(query_file), "-o", str(output_csv)]
    if debug:
        cmd.append("--debug")
    print(f"[run] {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    t1 = time.perf_counter()
    return t1 - t0


@dataclass
class Experiment:
    dbpedia_limit: int
    search_limit: int
    n_models: int


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fact-check Judge experiment runner")
    ap.add_argument("--config", default="config.ini")
    ap.add_argument("--query", default="queries/web/fact-check-judge.sparql")
    ap.add_argument("--slm-run", dest="slm_run", default=None)
    ap.add_argument("--out-dir", default="out/experiments-fcj")
    ap.add_argument("--dbpedia-limits", default="5,10", help="Ex: 3,5,10")
    ap.add_argument("--search-limits", default="1,3", help="Ex: 1,3,5")
    ap.add_argument(
        "--models",
        default="llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-20b",
        help="Liste de modèles base (dans l'ordre); on prendra les N premiers",
    )
    ap.add_argument("--n-models", default="1,3", help="Tailles N à tester, ex: 1,2,3")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    slm_bin = find_slm_run(args.slm_run)
    base_query_path = Path(args.query)
    if not base_query_path.exists():
        print(f"[error] Query not found: {base_query_path}", file=sys.stderr)
        sys.exit(1)

    base_query = read_text(base_query_path)
    out_dir = Path(args.out_dir)
    q_dir = out_dir / "queries"
    r_dir = out_dir / "results"
    q_dir.mkdir(parents=True, exist_ok=True)
    r_dir.mkdir(parents=True, exist_ok=True)

    db_limits = parse_int_list(args.dbpedia_limits)
    se_limits = parse_int_list(args.search_limits)
    models_base = [m.strip() for m in args.models.split(',') if m.strip()]
    n_models_list = parse_int_list(args.n_models)

    summary_rows: List[dict] = []

    for db_lim in db_limits:
        for se_lim in se_limits:
            for n_mod in n_models_list:
                if n_mod < 1 or n_mod > len(models_base):
                    print(f"[skip] n_models={n_mod} not in [1..{len(models_base)}]")
                    continue
                models = models_base[:n_mod]

                # Construire la requête spécifique
                q = base_query
                q = set_dbpedia_limit(q, db_lim)
                q = set_search_limit(q, se_lim)
                q = set_models(q, models)

                name = f"db{db_lim}_se{se_lim}_nm{n_mod}"
                q_path = q_dir / f"fcj_{name}.sparql"
                out_csv = r_dir / f"fcj_{name}.csv"
                write_text(q_path, q)

                # Exécuter et mesurer
                status = "ok"
                wall = None
                try:
                    wall = run_slm(slm_bin, args.config, q_path, out_csv, debug=args.debug)
                except subprocess.CalledProcessError as e:
                    print(f"[error] slm-run failed: {e}")
                    status = "fail"

                # Lire résultats si dispo
                total = confirmed = refuted = tie = 0
                if status == "ok" and out_csv.exists():
                    try:
                        df = pd.read_csv(out_csv)
                        total = len(df)
                        if total:
                            dec_col = "majorityDecision"
                            if dec_col in df.columns:
                                confirmed = int((df[dec_col] == "confirmed").sum())
                                refuted = int((df[dec_col] == "refuted").sum())
                                tie = int((df[dec_col] == "tie").sum())
                    except Exception as e:
                        print(f"[warn] Could not parse CSV {out_csv}: {e}")

                summary_rows.append(
                    {
                        "dbpedia_limit": db_lim,
                        "search_limit": se_lim,
                        "n_models": n_mod,
                        "models": ",".join(models),
                        "status": status,
                        "wall_time_sec": round(wall or 0.0, 3),
                        "total_rows": total,
                        "confirmed_rows": confirmed,
                        "refuted_rows": refuted,
                        "tie_rows": tie,
                        "results_csv": str(out_csv),
                        "query_file": str(q_path),
                    }
                )

    # Sauver le résumé
    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        if summary_rows:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        else:
            f.write("dbpedia_limit,search_limit,n_models,models,status,wall_time_sec,total_rows,confirmed_rows,refuted_rows,tie_rows,results_csv,query_file\n")

    print(f"[done] Summary written: {summary_csv} ({len(summary_rows)} runs)")


if __name__ == "__main__":
    main()
