#!/usr/bin/env python3

import spacy
from geopy.geocoders import Nominatim
import sys
import re
import csv
import time
import statistics
import json
import argparse
import math
import pandas as pd
from pathlib import Path
from tqdm import tqdm


###############################################
#  CONFIG
###############################################

DIST_THRESHOLD_KM = 30


###############################################
#  SPACY MODEL LOADING
###############################################

def load_spacy_model(model_name="en_core_web_trf"):
    """Load spaCy model, with fallback options"""
    try:
        print(f"⚙ Chargement du modèle spaCy '{model_name}'...")
        nlp = spacy.load(model_name)
        print(f"✓ Modèle '{model_name}' chargé avec succès")
        return nlp
    except OSError:
        print(f"⚠ Modèle '{model_name}' non trouvé. Tentative de téléchargement...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
        nlp = spacy.load(model_name)
        print(f"✓ Modèle '{model_name}' téléchargé et chargé")
        return nlp


###############################################
#  GEOLOCATION WITH SPACY
###############################################

def geotag(text, nlp):
    """Extract geographic entities from text using spaCy NER"""
    if not text or not text.strip():
        return None
    
    doc = nlp(text)
    
    if not doc.ents:
        return None

    label_to_keep = ["ORG", "FAC", "LOC", "GPE"]
    geo_ents = [ent.text for ent in doc.ents if ent.label_ in label_to_keep]
    
    geo_ents = list(dict.fromkeys(geo_ents))
    
    if not geo_ents:
        return None
    
    formated_location = ", ".join(geo_ents)
    
    return formated_location


###############################################
#  COORDINATES UTILITIES
###############################################

def get_coordinate(place):
    """Geocode a place name to coordinates"""
    app = Nominatim(user_agent="xml_geo_agent")
    try:
        location = app.geocode(place, timeout=10)
        if location:
            return {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "place_name": place
            }
        else:
            return None
    except Exception:
        return None


def check_coordinate(coord_string):
    """Validate coordinate string format"""
    if isinstance(coord_string, str) and re.match(r'^-?\d+\.?\d*,\s*-?\d+\.?\d*$', coord_string):
        try:
            lat_str, lon_str = coord_string.split(",")
            lat, lon = float(lat_str.strip()), float(lon_str.strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "place_name": coord_string
                }
        except:
            pass
    return None


###############################################
#  XML TEXT EXTRACTION
###############################################

def extract_text_from_xml(xml_block):
    """Extract meaningful text from BioSample XML block"""
    attributes_texts = []
    attribute_pattern = r'<Attribute[^>]*>(.*?)</Attribute>'
    attributes = re.findall(attribute_pattern, xml_block, re.DOTALL)
    attributes_texts.extend([attr.strip() for attr in attributes if attr.strip()])
    
    desc_pattern = r'<Description>(.*?)</Description>'
    desc_match = re.search(desc_pattern, xml_block, re.DOTALL)
    if desc_match:
        attributes_texts.append(desc_match.group(1).strip())
    
    title_pattern = r'<Title>(.*?)</Title>'
    title_match = re.search(title_pattern, xml_block, re.DOTALL)
    if title_match:
        attributes_texts.append(title_match.group(1).strip())
    
    full_text = " ".join(attributes_texts)
    full_text = re.sub(r'&[a-z]+;', ' ', full_text)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    
    return full_text


def extract_coordinate_from_xml(xml_block):
    """Directly extract coordinates if explicitly present in XML"""
    coord_patterns = [
        r'<Attribute[^>]*attribute_name="(?:lat_lon|latitude_longitude|coordinates)"[^>]*>(.*?)</Attribute>',
        r'<Attribute[^>]*>(\-?\d+\.?\d*,\s*\-?\d+\.?\d*)</Attribute>'
    ]
    
    for pattern in coord_patterns:
        match = re.search(pattern, xml_block, re.IGNORECASE)
        if match:
            coord_str = match.group(1).strip()
            result = check_coordinate(coord_str)
            if result:
                return result
    
    return None


###############################################
#  PROCESS WITH SPACY
###############################################

def process_biosample_with_spacy(xml_block, nlp, log_file=None):
    """Process one BioSample block with spaCy"""
    start_time = time.time()
    
    coord_result = extract_coordinate_from_xml(xml_block)
    if coord_result:
        exec_time = time.time() - start_time
        if log_file:
            log_file.write(f"DIRECT COORDINATES FOUND: {coord_result}\n")
        return coord_result, exec_time
    
    text = extract_text_from_xml(xml_block)
    
    if log_file:
        log_file.write(f"EXTRACTED TEXT: {text[:300]}...\n")
    
    location_string = geotag(text, nlp)
    
    if log_file:
        log_file.write(f"SPACY ENTITIES: {location_string}\n")
    
    if not location_string:
        exec_time = time.time() - start_time
        return {"latitude": None, "longitude": None, "place_name": None}, exec_time
    
    geo_result = get_coordinate(location_string)
    
    exec_time = time.time() - start_time
    
    if geo_result:
        if log_file:
            log_file.write(f"GEOCODED: {geo_result}\n")
        return geo_result, exec_time
    else:
        if log_file:
            log_file.write(f"GEOCODING FAILED for: {location_string}\n")
        return {"latitude": None, "longitude": None, "place_name": location_string}, exec_time


###############################################
#  PROCESS BIOSAMPLES
###############################################

def process_biosamples(biosamples_filename, nlp, output_file, log_file):
    """Process all BioSample blocks from XML file"""
    
    with open(biosamples_filename, "r", encoding="utf-8") as f:
        xml_data = f.read()

    biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    total = len(biosample_blocks)

    print(f"📊 Found {total} BioSample blocks\n")

    exec_times = []
    success_count = 0
    results = []

    for block in tqdm(biosample_blocks, desc="Processing samples", unit="sample"):
        
        acc = re.search(r'accession="([^"]+)"', block)
        accession = acc.group(1) if acc else "UNKNOWN"

        result, exec_time = process_biosample_with_spacy(block, nlp, log_file)
        exec_times.append(exec_time)

        log_file.write("\n" + "="*60 + "\n")
        log_file.write(f"ACCESSION: {accession}\n")
        log_file.write(f"EXECUTION TIME: {exec_time:.2f}s\n")
        log_file.write(f"RESULT: {json.dumps(result, indent=2, ensure_ascii=False)}\n")
        log_file.write("="*60 + "\n")
        log_file.flush()

        if result.get("latitude") is not None:
            success_count += 1
        
        row = {
            "accession": accession,
            "latitude": result.get("latitude") or "",
            "longitude": result.get("longitude") or "",
            "place": result.get("place_name") or "",
            "execution_time": f"{exec_time:.2f}"
        }
        
        output_file.writerow(row)
        results.append(row)

    print(f"\n✓ Traitement terminé:")
    print(f"  - {success_count}/{total} échantillons géolocalisés")
    print(f"  - Temps moyen par échantillon: {statistics.mean(exec_times):.2f}s")
    print(f"  - Temps total: {sum(exec_times):.1f}s\n")
    
    return results


###############################################
#  EVALUATION
###############################################

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km"""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2

    return 2 * R * math.asin(math.sqrt(a))


COORD_RE = re.compile(r"-?\d+(\.\d+)?\s*,\s*-?\d+(\.\d+)?")

def load_expected_answers(expected_file):
    """Parse expected answers from TSV file"""
    if not Path(expected_file).exists():
        print(f"⚠ Fichier de référence non trouvé: {expected_file}")
        print("  Évaluation ignorée.")
        return None
    
    expected = {}

    with open(expected_file, "r", encoding="utf-8") as f:
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


def evaluate_results(results, expected):
    """Evaluate predictions against expected answers"""
    eval_rows = []
    
    for row in results:
        acc = row["accession"]
        
        pred_lat = None if row["latitude"] == "" else float(row["latitude"])
        pred_lon = None if row["longitude"] == "" else float(row["longitude"])
        exec_time = float(row["execution_time"])
        
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
        
        eval_rows.append({
            "accession": acc,
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "exp_lat": exp["latitude"],
            "exp_lon": exp["longitude"],
            "distance_km": dist_km,
            "execution_time": exec_time,
            "category": category
        })
    
    return pd.DataFrame(eval_rows)


def compute_metrics(df_eval):
    """Calculate all metrics"""
    tp = (df_eval["category"] == "TP").sum()
    tn = (df_eval["category"] == "TN").sum()
    fp = (df_eval["category"] == "FP").sum()
    fn = (df_eval["category"] == "FN").sum()
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    times = df_eval["execution_time"]
    tp_distances = df_eval[df_eval["category"] == "TP"]["distance_km"].dropna()
    
    metrics = {
        "total_samples": len(df_eval),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "mean_time_s": round(times.mean(), 3),
        "median_time_s": round(times.median(), 3),
        "total_time_s": round(times.sum(), 3),
        "mean_distance_km": round(tp_distances.mean(), 3) if len(tp_distances) > 0 else None,
        "median_distance_km": round(tp_distances.median(), 3) if len(tp_distances) > 0 else None,
        "max_distance_km": round(tp_distances.max(), 3) if len(tp_distances) > 0 else None
    }
    
    return metrics


def print_evaluation_results(metrics, df_eval):
    """Print formatted evaluation results"""
    print("\n" + "="*80)
    print("📊 RÉSULTATS D'ÉVALUATION")
    print("="*80)
    
    print(f"\n🎯 CLASSIFICATION (seuil de distance: {DIST_THRESHOLD_KM} km)")
    print(f"  • True Positives (TP):  {metrics['tp']:3d}  (localisations correctes)")
    print(f"  • True Negatives (TN):  {metrics['tn']:3d}  (absence correctement détectée)")
    print(f"  • False Positives (FP): {metrics['fp']:3d}  (fausses détections)")
    print(f"  • False Negatives (FN): {metrics['fn']:3d}  (localisations manquées)")
    
    print(f"\n📈 MÉTRIQUES DE QUALITÉ")
    print(f"  • Accuracy:    {metrics['accuracy']:.1%}  (précision globale)")
    print(f"  • Precision:   {metrics['precision']:.1%}  (fiabilité des détections)")
    print(f"  • Recall:      {metrics['recall']:.1%}  (sensibilité)")
    print(f"  • Specificity: {metrics['specificity']:.1%}  (taux de vrais négatifs)")
    print(f"  • F1-Score:    {metrics['f1_score']:.1%}  (score combiné)")
    
    print(f"\n⏱️  PERFORMANCE")
    print(f"  • Temps moyen:   {metrics['mean_time_s']:.3f}s par échantillon")
    print(f"  • Temps médian:  {metrics['median_time_s']:.3f}s")
    print(f"  • Temps total:   {metrics['total_time_s']:.1f}s")
    
    # if metrics['mean_distance_km'] is not None:
    #     print(f"\n📍 PRÉCISION GÉOGRAPHIQUE (pour les TP uniquement)")
    #     print(f"  • Distance moyenne:  {metrics['mean_distance_km']:.2f} km")
    #     print(f"  • Distance médiane:  {metrics['median_distance_km']:.2f} km")
    #     print(f"  • Distance max:      {metrics['max_distance_km']:.2f} km")
    
    # print("\n" + "="*80)
    
    # # Show some examples
    # print("\n💡 EXEMPLES DE PRÉDICTIONS")
    # print("-"*80)
    
    # tp_examples = df_eval[df_eval["category"] == "TP"].head(3)
    # if len(tp_examples) > 0:
    #     print("\n✓ Exemples de prédictions correctes (TP):")
    #     for _, row in tp_examples.iterrows():
    #         print(f"  {row['accession']}: erreur de {row['distance_km']:.2f} km")
    
    # fp_examples = df_eval[df_eval["category"] == "FP"].head(3)
    # if len(fp_examples) > 0:
    #     print("\n✗ Exemples de fausses détections (FP):")
    #     for _, row in fp_examples.iterrows():
    #         if row['distance_km'] is not None:
    #             print(f"  {row['accession']}: erreur de {row['distance_km']:.2f} km (> {DIST_THRESHOLD_KM} km)")
    #         else:
    #             print(f"  {row['accession']}: localisation détectée mais non attendue")
    
    # fn_examples = df_eval[df_eval["category"] == "FN"].head(3)
    # if len(fn_examples) > 0:
    #     print("\n✗ Exemples de localisations manquées (FN):")
    #     for _, row in fn_examples.iterrows():
    #         print(f"  {row['accession']}: attendu ({row['exp_lat']}, {row['exp_lon']})")
    
    # print("\n" + "="*80)


###############################################
#  MAIN
###############################################

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Process biosamples XML file with spaCy for location extraction"
    )
    parser.add_argument("--input", type=str, required=True,
                       help="Input XML file with BioSample data")
    parser.add_argument("--output", type=str, default="biosample_results_spacy.csv",
                       help="Output CSV file")
    parser.add_argument("--model", type=str, default="en_core_web_trf",
                       help="spaCy model to use (default: en_core_web_trf)")
    parser.add_argument("--expected", type=str, default=None,
                       help="Expected answers TSV file for evaluation")

    args = parser.parse_args()

    # Auto-detect expected file if not provided
    if args.expected is None:
        input_path = Path(args.input)
        expected_path = input_path.parent / f"{input_path.stem}_answer.tsv"
        if expected_path.exists():
            args.expected = str(expected_path)
            print(f"📋 Fichier de référence détecté: {args.expected}\n")

    # Load spaCy model
    nlp = load_spacy_model(args.model)

    # Open log file
    log_filename = args.output + ".log"
    log_file = open(log_filename, "w", encoding="utf-8")
    log_file.write(f"SPACY MODEL: {args.model}\n")
    log_file.write(f"INPUT FILE: {args.input}\n")
    log_file.write(f"OUTPUT FILE: {args.output}\n")
    log_file.write("="*60 + "\n\n")

    # Process biosamples
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["accession", "latitude", "longitude", "place", "execution_time"]
        )
        writer.writeheader()
        
        results = process_biosamples(args.input, nlp, writer, log_file)

    log_file.close()
    
    print(f"📄 Résultats sauvegardés dans: {args.output}")
    print(f"📄 Logs sauvegardés dans: {log_filename}")

    # Evaluate if expected file is available
    if args.expected:
        expected = load_expected_answers(args.expected)
        
        if expected:
            df_eval = evaluate_results(results, expected)
            
            # Save detailed evaluation
            eval_output = args.output.replace(".csv", "_evaluation.csv")
            df_eval.to_csv(eval_output, index=False)
            
            # Compute and display metrics
            metrics = compute_metrics(df_eval)
            
            # Save metrics as JSON
            metrics_output = args.output.replace(".csv", "_metrics.json")
            with open(metrics_output, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Display results
            print_evaluation_results(metrics, df_eval)
            
            print(f"\n📄 Évaluation détaillée: {eval_output}")
            print(f"📄 Métriques JSON: {metrics_output}")
    else:
        print("\n⚠ Pas de fichier de référence fourni. Évaluation ignorée.")
        print("  Utilisez --expected pour spécifier un fichier de référence.")