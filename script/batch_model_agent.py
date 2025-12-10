#!/usr/bin/env python3
import subprocess
import csv
import os
import json
import math
import pandas as pd
from pathlib import Path

########################################
# CONFIG
########################################

MODELS = [
    "gpt-oss:20b",
    "llama3.1:8b",
    "mistral-nemo:12b"
]

LLM_SCRIPT = "script/llm_agent_api.py"
DEFAULT_INPUT = "data_test/Dataset_test_biosample.xml"
EXPECTED_FILE = "data_test/Dataset_test_biosample_answer.tsv"
OUTPUT_DIR = "results_llm"


########################################
# LAUNCH LLM ON ALL MODELS
########################################

def run_all_models():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    for model in MODELS:
        output_file = f"{OUTPUT_DIR}/{model.replace(':', '_')}.csv"

        cmd = [
            "python3",
            LLM_SCRIPT,
            "--input", DEFAULT_INPUT,
            "--model", model,
            "--output", output_file
        ]

        print(f"\n=== RUNNING MODEL: {model} ===")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)
        print(f"→ Output saved to {output_file}")


########################################
# GEOGRAPHICAL DISTANCE (Haversine)
########################################

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(d_lam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


########################################
# LOAD EXPECTED ANSWERS
########################################

def load_expected_answers():
    expected = {}
    with open(EXPECTED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            acc, place, coords = line.strip().split("\t")
            if coords != "NA":
                lat, lon = coords.split(",")
                expected[acc] = {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "place": place
                }
            else:
                expected[acc] = {
                    "latitude": None,
                    "longitude": None,
                    "place": None
                }
    return expected


########################################
# EVALUATION OF ONE MODEL RESULT FILE
########################################

def evaluate_result_file(csv_file, expected):
    df = pd.read_csv(csv_file)
    rows = []

    for _, row in df.iterrows():
        acc = row["accession"]
        pred_lat = row["latitude"] if row["latitude"] != "" else None
        pred_lon = row["longitude"] if row["longitude"] != "" else None

        exp = expected.get(acc, None)
        if exp is None:
            continue

        category = "UNK"
        dist_km = None

        if exp["latitude"] is None:
            # expected NA
            if pred_lat is None:
                category = "TN"
            else:
                category = "FP"
        else:
            # expected coords
            if pred_lat is None:
                category = "FN"
            else:
                dist_km = haversine(float(pred_lat), float(pred_lon),
                                    exp["latitude"], exp["longitude"])

                if dist_km <= 30:  # threshold
                    category = "TP"
                else:
                    category = "FP"

        rows.append({
            "accession": acc,
            "model": row["model"],
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "exp_lat": exp["latitude"],
            "exp_lon": exp["longitude"],
            "distance_km": dist_km if dist_km else "",
            "category": category
        })

    return rows


########################################
# MAIN EXECUTION
########################################

def main():
    print("\n=== Step 1: Running all models ===")
    run_all_models()

    print("\n=== Step 2: Loading expected answers ===")
    expected = load_expected_answers()

    print("\n=== Step 3: Evaluating results ===")
    final_results = []

    for model in MODELS:
        csv_file = f"{OUTPUT_DIR}/{model.replace(':', '_')}.csv"
        print(f"Evaluating: {csv_file}")

        rows = evaluate_result_file(csv_file, expected)
        final_results.extend(rows)

    print("\n=== Step 4: Saving evaluation ===")
    out_file = f"{OUTPUT_DIR}/evaluation_results.csv"
    pd.DataFrame(final_results).to_csv(out_file, index=False)

    print(f"\n✔ Evaluation saved to {out_file}")


if __name__ == "__main__":
    main()
