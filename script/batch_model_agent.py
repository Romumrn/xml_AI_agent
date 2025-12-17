#!/usr/bin/env python3
import subprocess
import math
import pandas as pd
from pathlib import Path
import re

########################################
# CONFIG
########################################

MODELS = [
    "cogito:8b",
    "qwen3:8b",
    "granite4:3b",
    "gpt-oss:20b",
    "llama3.1:8b",
    "mistral-nemo:12b"
]

LLM_SCRIPT = "script/llm_agent_api.py"
DEFAULT_INPUT = "data_test/Dataset_test_biosample.xml"
EXPECTED_FILE = "data_test/Dataset_test_biosample_answer.tsv"
OUTPUT_DIR = "results_llm"

DIST_THRESHOLD_KM = 30

########################################
# RUN MODELS
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
        subprocess.run(cmd, check=True)
        print(f"→ Output saved to {output_file}")

########################################
# HAVERSINE
########################################

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2

    return 2 * R * math.asin(math.sqrt(a))

########################################
# LOAD EXPECTED ANSWERS (ROBUST)
########################################

COORD_RE = re.compile(r"-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?")

def load_expected_answers():
    expected = {}

    with open(EXPECTED_FILE, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            acc = line.split()[0]
            coord_match = COORD_RE.search(line)

            if coord_match:
                coords = coord_match.group()
                place = line[len(acc):coord_match.start()].strip()
            else:
                coords = "NA"
                place = line[len(acc):].strip()

            if place.upper() == "NA" or place == "":
                place = None

            if coords.upper() == "NA":
                expected[acc] = {"latitude": None, "longitude": None}
                continue

            try:
                lat, lon = coords.split(",")
                expected[acc] = {
                    "latitude": float(lat),
                    "longitude": float(lon)
                }
            except Exception:
                expected[acc] = {"latitude": None, "longitude": None}

    return expected

########################################
# EVALUATE PREDICTIONS
########################################

def evaluate_result_file(csv_file, expected):
    df = pd.read_csv(csv_file)
    rows = []

    for _, row in df.iterrows():
        acc = row["accession"]
        model = row["model"]

        pred_lat = None if pd.isna(row["latitude"]) else float(row["latitude"])
        pred_lon = None if pd.isna(row["longitude"]) else float(row["longitude"])
        exec_time = None if pd.isna(row["execution_time"]) else float(row["execution_time"])

        exp = expected.get(acc)
        if exp is None:
            continue

        dist_km = None

        if exp["latitude"] is None:
            category = "TN" if pred_lat is None else "FP"
        else:
            if pred_lat is None:
                category = "FN"
            else:
                dist_km = haversine(
                    pred_lat, pred_lon,
                    exp["latitude"], exp["longitude"]
                )
                category = "TP" if dist_km <= DIST_THRESHOLD_KM else "FP"

        rows.append({
            "accession": acc,
            "model": model,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "exp_lat": exp["latitude"],
            "exp_lon": exp["longitude"],
            "distance_km": dist_km,
            "execution_time": exec_time,
            "category": category
        })

    return rows

########################################
# METRICS (QUALITY)
########################################

def compute_quality_metrics(df):
    rows = []

    for model in MODELS:
        sub = df[df["model"] == model]

        tp = (sub["category"] == "TP").sum()
        tn = (sub["category"] == "TN").sum()
        fp = (sub["category"] == "FP").sum()
        fn = (sub["category"] == "FN").sum()

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        rows.append({
            "model": model,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4)
        })

    return pd.DataFrame(rows)

########################################
# METRICS (TIME)
########################################

def compute_time_metrics(df):
    rows = []

    for model in MODELS:
        times = pd.to_numeric(
            df[df["model"] == model]["execution_time"],
            errors="coerce"
        ).dropna()

        if times.empty:
            continue

        rows.append({
            "model": model,
            "mean_time_s": round(times.mean(), 3),
            "median_time_s": round(times.median(), 3),
            "p95_time_s": round(times.quantile(0.95), 3),
            "min_time_s": round(times.min(), 3),
            "max_time_s": round(times.max(), 3)
        })

    return pd.DataFrame(rows)

########################################
# MAIN
########################################

def main():
    print("\n=== STEP 1: RUN MODELS ===")
    run_all_models()

    print("\n=== STEP 2: LOAD EXPECTED ===")
    expected = load_expected_answers()

    print("\n=== STEP 3: EVALUATE ===")
    all_rows = []

    for model in MODELS:
        csv_file = f"{OUTPUT_DIR}/{model.replace(':', '_')}.csv"
        print(f"Evaluating {csv_file}")
        all_rows.extend(evaluate_result_file(csv_file, expected))

    df_eval = pd.DataFrame(all_rows)
    df_eval.to_csv(f"{OUTPUT_DIR}/evaluation_results.csv", index=False)

    print("\n=== STEP 4: METRICS ===")
    quality_df = compute_quality_metrics(df_eval)
    time_df = compute_time_metrics(df_eval)

    quality_df.to_csv(f"{OUTPUT_DIR}/model_quality_metrics.csv", index=False)
    time_df.to_csv(f"{OUTPUT_DIR}/model_time_metrics.csv", index=False)

    print("\n===== QUALITY METRICS =====")
    print(quality_df.to_string(index=False))

    print("\n===== TIME METRICS (seconds) =====")
    print(time_df.to_string(index=False))

    print("\n✔ All results saved in results_llm/")

if __name__ == "__main__":
    main()
