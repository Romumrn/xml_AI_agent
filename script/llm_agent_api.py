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
from tqdm import tqdm


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
            else:
                print( "WRONG COORD ? ", coord)
        except:
            pass
    return "NA"



###############################################
#  JSON NORMALIZATION
###############################################

def select_best_json(text):
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
    if res is None:
        return {"latitude": None, "longitude": None, "place_name": None}

    if isinstance(res, str) and res.strip().lower() == "na":
        return {"latitude": None, "longitude": None, "place_name": None}

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

    if isinstance(res, str) and "," in res:
        try:
            lat, lon = res.split(",", 1)
            return {"latitude": float(lat), "longitude": float(lon), "place_name": res}
        except:
            pass

    if isinstance(res, str):
        return {"latitude": None, "longitude": None, "place_name": res.strip()}

    return {"latitude": None, "longitude": None, "place_name": None}



###############################################
#  LLM CALL WORKER
###############################################

def _ollama_call_worker(model, block_xml):

    prompt = f"""
----------------------------------------
You are a strict information extraction system.

You MUST follow the procedure exactly.
Do NOT guess.
Do NOT infer missing data.
Do NOT fabricate coordinates.

----------------------------------------
INPUT XML:
{block_xml}
----------------------------------------

PROCEDURE (MANDATORY):

STEP 1 — DETECTION
Scan the XML and detect geographic information ONLY:
- Place names (city, region, country, landmark)
- Coordinates (latitude/longitude)

If NOTHING geographic is found:
→ Return JSON with place_name="NA" and latitude=null, longitude=null
→ STOP

STEP 2 — COORDINATES HANDLING
If coordinates are explicitly present in the XML:
→ Format them as "lat,lon"
→ CALL check_coordinate with the coordinate
→ Use ONLY validated coordinates in the final JSON

STEP 3 — PLACE NAME HANDLING
If a place name is found WITHOUT coordinates:
→ CALL get_coordinate using the exact place name string
→ Use the returned coordinates
→ DO NOT invent coordinates

RULES:
- Use tools whenever required by the step
- Never return coordinates without a tool call
- Never call both tools for the same data
- Return ONE most specific location only

FINAL OUTPUT FORMAT (JSON ONLY):
{{
  "place_name": "string",
  "latitude": float|null,
  "longitude": float|null
}}
NO explanation.
NO extra text.

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

    if response.message.tool_calls:
        for call in response.message.tool_calls:
            fname = call.function.name
            args = call.function.arguments

            if fname == "get_coordinate":
                outputs.append(get_coordinate(args["place_name"]))

            elif fname == "check_coordinate":
                outputs.append(check_coordinate(args["coordinate"]))

        return outputs

    return [response.message["content"]]



###############################################
#  PROCESS BIOSAMPLES
###############################################

def process_biosamples(biosamples_filename, model, writer_output_file, log_file):
    
    with open(biosamples_filename, "r", encoding="utf-8") as f:
        xml_data = f.read()

    biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    total = len(biosample_blocks)

    print(f"Found {total} BioSample blocks. Processing with model: {model}\n")

    exec_times = []

    for block in tqdm(biosample_blocks, desc="Processing samples", unit="sample"):
        start = time.time()

        acc = re.search(r'accession="([^"]+)"', block)
        accession = acc.group(1) if acc else "UNKNOWN"

        raw = ask_agent(block, model)

        # ------ LOGGING ------
        log_file.write("\n============================================\n")
        log_file.write(f"ACCESSION: {accession}\n")
        log_file.write(f"MODEL: {model}\n")
        log_file.write("RAW MODEL OUTPUT:\n")
        log_file.write(json.dumps(raw, indent=2, ensure_ascii=False))
        log_file.write("\n============================================\n")
        log_file.flush()
        # ----------------------

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

    print(f"\nMoyenne des temps d'appel LLM : {statistics.mean(exec_times):.2f}s")
    print(f"Temps total approx : {sum(exec_times):.1f}s\n")



###############################################
#  MAIN
###############################################

if __name__ == "__main__":
    mp.freeze_support()

    parser = argparse.ArgumentParser(
        description="Process biosamples XML file with LLM for location extraction"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="biosample_results.csv")
    parser.add_argument("--model", type=str, default="llama3.1:8b")

    args = parser.parse_args()

    log_filename = args.output + ".log"
    log_file = open(log_filename, "a", encoding="utf-8")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["accession", "model", "latitude", "longitude", "place", "execution_time"]
        )
        writer.writeheader()

        if check_and_pull_model(args.model):
            process_biosamples(args.input, args.model, writer, log_file)
        else:
            print("✗ Impossible d'utiliser ce modèle.")

    log_file.close()
