#!/usr/bin/env python3
"""
Automate the execution of the "extract" SPARQL query for multiple configurations,
measure execution time, and evaluate FAISS/LLM metrics.

Usage:
  python3 scripts/run_experiment.py --configurations configs.json --out-dir out/

Expected structure of `configs.json`:
[
  {
    "name": "gpt-oss-20b",
    "model": "openai/gpt-oss-20b"
  },
  {
    "name": "llama-3.1-8b",
    "model": "llama-3.1-8b-instant"
  }
]

Output:
- Results are saved in `out/<configuration_name>/`.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

def run_sparql_query(config, out_dir):
    """Run the SPARQL query for a given configuration and measure execution time."""
    query_file = "queries/cultural/mixed-event-extract.sparql"
    output_csv = out_dir / "search.csv"
    results_file = out_dir / "execution_time.txt"
    executed_query_file = out_dir / "executed_query.sparql"  # File to save the executed query
    raw_results_file = out_dir / "raw_results.txt"  # File to save raw query results

    # Modify the query dynamically to use the specified model
    with open(query_file, "r") as f:
        query = f.read()
    query = query.replace("\"openai/gpt-oss-20b\"", f'"{config["model"]}"')

    # Write the modified query to a temporary file
    temp_query_file = out_dir / "temp_query.sparql"
    with open(temp_query_file, "w") as f:
        f.write(query)

    # Save the executed query for reference
    with open(executed_query_file, "w") as f:
        f.write(query)

    # Run the query and measure execution time
    start_time = time.time()
    with open(raw_results_file, "w") as raw_output:
        subprocess.run([
            "slm-run", "--config", "config.ini", "-f", str(temp_query_file), "-o", str(output_csv)
        ], check=True, stdout=raw_output, stderr=subprocess.STDOUT)
    end_time = time.time()

    # Save execution time
    with open(results_file, "w") as f:
        f.write(f"Execution time: {end_time - start_time:.2f} seconds\n")

    # Clean up temporary query file
    temp_query_file.unlink()

def evaluate_results(out_dir):
    """Evaluate the results using the FAISS/LLM evaluation script."""
    input_csv = out_dir / "search.csv"
    eval_json = out_dir / "eval_results.json"

    subprocess.run([
        "python3", "scripts/evaluate_faiss_llm.py",
        "--input", str(input_csv), "--out-json", str(eval_json)
    ], check=True)

def main():
    parser = argparse.ArgumentParser(description="Run SPARQL experiments for multiple configurations.")
    parser.add_argument("--configurations", required=True, help="Path to the JSON file with configurations.")
    parser.add_argument("--out-dir", required=True, help="Output directory for results.")
    args = parser.parse_args()

    # Load configurations
    with open(args.configurations, "r") as f:
        configs = json.load(f)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        print(f"Running configuration: {config['name']}")
        config_out_dir = out_dir / config['name']
        config_out_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Run SPARQL query
        run_sparql_query(config, config_out_dir)

        # Step 2: Evaluate results
        evaluate_results(config_out_dir)

        print(f"Results saved in: {config_out_dir}")

if __name__ == "__main__":
    main()