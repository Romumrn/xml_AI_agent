#!/usr/bin/env python3

import ollama
from geopy.geocoders import Nominatim
import sys
import re
import csv
import time
import statistics
import subprocess
import json
import multiprocessing as mp
import argparse

###############################################
#  CHECK & INSTALL MODEL
###############################################

def check_and_pull_model(model):
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        model_base = model.split(":")[0]

        if model_base in result.stdout or model in result.stdout:
            print(f"✓ Modèle '{model}' déjà installé")
            return True
        else:
            print(f"⚠ Modèle '{model}' non trouvé, téléchargement…")
            subprocess.run(["ollama", "pull", model], check=True)
            print(f"✓ Modèle '{model}' téléchargé avec succès")
            return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors du téléchargement du modèle: {e}")
        return False
    except FileNotFoundError:
        print("✗ Ollama n'est pas installé.")
        return False



###############################################
#  COORDINATES UTILITIES
###############################################

def get_coordinate(place):
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
            return "NA"
    except Exception:
        return "NA"


def check_coordinate(coord):
    if isinstance(coord, str) and re.match(r'^-?\d+\.?\d*,\s*-?\d+\.?\d*$', coord):
        try:
            lat_str, lon_str = coord.split(",")
            lat, lon = float(lat_str), float(lon_str)
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {
                    "latitude": lat,
                    "longitude": lon,
                    "place_name": coord
                }
        except:
            pass
    return "NA"



###############################################
#  JSON NORMALIZATION
###############################################

def select_best_json(text):
    """
    Trouve le JSON le plus gros + valide dans une réponse brute.
    Utile si le modèle renvoie plusieurs {} ou mélange texte/JSON.
    """
    json_candidates = re.findall(r'\{.*?\}', text, flags=re.DOTALL)
    best = None
    best_len = 0

    for cand in json_candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and len(cand) > best_len:
                best = obj
                best_len = len(cand)
        except:
            continue
    return best


def normalize_llm_result(res):
    """
    Normalise toutes formes de réponses en un dict cohérent.
    """

    if res is None:
        return {"latitude": None, "longitude": None, "place_name": None}

    if isinstance(res, str) and res.strip().lower() == "na":
        return {"latitude": None, "longitude": None, "place_name": None}

    # Essayer d'extraire JSON
    if isinstance(res, str):
        json_obj = select_best_json(res)
        if json_obj:
            return {
                "latitude": json_obj.get("latitude"),
                "longitude": json_obj.get("longitude"),
                "place_name": json_obj.get("place_name")
            }

    if isinstance(res, dict):
        return {
            "latitude": res.get("latitude"),
            "longitude": res.get("longitude"),
            "place_name": res.get("place_name") or res.get("LLM_place_found")
        }

    # lat,lon brut
    if isinstance(res, str) and "," in res:
        try:
            lat, lon = res.split(",", 1)
            return {"latitude": float(lat), "longitude": float(lon), "place_name": res}
        except:
            pass

    # nom de lieu simple
    if isinstance(res, str):
        return {"latitude": None, "longitude": None, "place_name": res.strip()}

    return {"latitude": None, "longitude": None, "place_name": None}



###############################################
#  LLM CALL WORKER
###############################################

def _ollama_call_worker(model, block_xml):

    # Prompt APRÈS le XML = meilleur parsing (TESTÉ)
    prompt = f"""
XML BLOCK:
----------------------------------------
{block_xml}
----------------------------------------

TASK:
Extract ONLY geographic location information from this XML block, and use the function if needed 
Or return NA if no place name found.

EXTRACT:
✓ City, region, country names
✓ Latitude/longitude coordinates
✓ Geographic landmarks

IGNORE:
✗ Species names, organism information
✗ Sample types, body sites
✗ Project names, study information

OUTPUT RULES:
- Return ONLY the most specific geographic identifier
- For coordinates: use decimal format (e.g., "40.7128,-74.0060")
- For place names: use specific format (e.g., "City, COUNTRY")
- If no geographic information: return "NA"
- If you find coordinates, use check_coordinate to validate them
- If you find place names, use get_coordinate to get coordinates

YOU MUST RETURN STRICT JSON IN THIS FORMAT:
{{
"place_name": "...",
"latitude": float|null,
"longitude": float|null
}}
Return JSON ONLY. No explanation.
"""

    return ollama.chat(
        model=model,
        options={"temperature": 0},
        messages=[{"role": "user", "content": prompt}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_coordinate",
                    "description": "Geocode a place name",
                    "parameters": {
                        "type": "object",
                        "properties": {"place_name": {"type": "string"}},
                        "required": ["place_name"]
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_coordinate",
                    "description": "Validate a coordinate pair",
                    "parameters": {
                        "type": "object",
                        "properties": {"coordinate": {"type": "string"}},
                        "required": ["coordinate"]
                    },
                }
            }
        ]
    )



###############################################
#  ASK AGENT
###############################################

def ask_agent(block_xml, model, timeout=60):

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        pending = pool.apply_async(_ollama_call_worker, (model, block_xml))

        try:
            response = pending.get(timeout=timeout)
        except mp.TimeoutError:
            print(f"[TIMEOUT] modèle {model}")
            return ["NA"]

    outputs = []

    # Tool calls
    if response.message.tool_calls:
        for call in response.message.tool_calls:
            fname = call.function.name
            args = call.function.arguments

            if fname == "get_coordinate":
                print( "get coords", args["place_name"] )
                outputs.append(get_coordinate(args["place_name"]))

            elif fname == "check_coordinate":
                outputs.append(check_coordinate(args["coordinate"]))

        return outputs

    # Sinon texte brut
    return [response.message["content"]]



###############################################
#  PROCESS BIOSAMPLES
###############################################

def process_biosamples(biosamples_filename, model, writer_output_file):

    with open(biosamples_filename, "r", encoding="utf-8") as f:
        xml_data = f.read()

    biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    print(f"Found {len(biosample_blocks)} BioSample blocks.")

    exec_times = []

    for i, block in enumerate(biosample_blocks, start=1):
        start = time.time()

        acc = re.search(r'accession="([^"]+)"', block)
        accession = acc.group(1) if acc else f"UNKNOWN_{i}"

        raw = ask_agent(block, model)
        print( raw)
        final = [normalize_llm_result(r) for r in raw]

        end = time.time()
        t = end - start
        exec_times.append(t)

        for f in final:
            writer_output_file.writerow({
                "accession": accession,
                "model": model,
                "latitude": f["latitude"] or "",
                "longitude": f["longitude"] or "",
                "place": f["place_name"] or "",
                "execution_time": f"{t:.2f}"
            })

        print(f"{i}/{len(biosample_blocks)}  OK  ({t:.2f}s)")

    print(f"\nMoyenne des temps : {statistics.mean(exec_times):.2f}s")



###############################################
#  MAIN
###############################################

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(
        description="Process biosamples XML file with LLM for location extraction"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input XML file containing biosample data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="biosample_results.csv",
        help="Path to the output CSV file (default: biosample_results.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="Model to use for processing (default: llama3.1:8b)"
    )

    args = parser.parse_args()

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["accession", "model", "latitude", "longitude", "place", "execution_time"]
        )
        writer.writeheader()

        if check_and_pull_model(args.model):
            process_biosamples(args.input, args.model, writer)
        else:
            print("✗ Impossible d'utiliser ce modèle.")