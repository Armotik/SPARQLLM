# Experimental Workflow: Generate, Search, and Evaluate Metrics

This document describes the experimental workflow for generating synthetic cultural event data, searching through the FAISS index, and evaluating the retrieval and extraction metrics.

```bash
 # Quick commands (run in this order):
 slm-run --config config.ini -f queries/cultural/generate_cultural_event_page.sparql -o cultural-events
 rm data/cultural_events_faiss.*
 python3 scripts/run_experiment.py --configurations configs.json --out-dir out/
 python3 scripts/plot_results.py --input-dir out/ --output-dir plots/
```

# Experimental Workflow: Generate, Search, and Evaluate Metrics

This document describes the experimental workflow for generating synthetic cultural event data, searching through the FAISS index, and evaluating the retrieval and extraction metrics.

## 1. Generate Synthetic Data

The first step is to generate synthetic cultural event data using a SPARQL query and an LLM. This step creates JSON-LD representations of events for various European capitals and cultural event categories.

### Command:
```bash
slm-run --config config.ini -f queries/cultural/generate_cultural_event_page.sparql -o cultural-events
```

### Output:

### Key Details:


## 2. Search the FAISS Index

The second step is to search the FAISS index for relevant events based on a query for each capital. The FAISS provider retrieves the top-k results for each query.

### Quick prep (remove old FAISS index files):
```bash
rm data/cultural_events_faiss.*
```

### Run the experiment (build indexes, search and extract):
```bash
python3 scripts/run_experiment.py --configurations configs.json --out-dir out/
```

### Output:

### Key Details:


## 3. Evaluate Metrics

The final step is to evaluate the performance of the retrieval (FAISS) and extraction (LLM) processes. Metrics include Recall@k, MRR, and extraction quality for fields like `venue`, `address`, `date`, and `capacity`.

### Generate plots from experiment outputs:
```bash
python3 scripts/plot_results.py --input-dir out/ --output-dir plots/
```

### Output:

### Key Metrics:
  - Recall@k (e.g., k=1, 3, 5, 10)
  - Mean Reciprocal Rank (MRR)
  - Exact match for `date` and `capacity`
  - Fuzzy similarity for `venue` and `address`


## Summary

This workflow enables the generation of synthetic cultural event data, retrieval of relevant events using FAISS, and evaluation of the retrieval and extraction quality. Adjustments to the prompts, FAISS index, or evaluation thresholds can further refine the results.