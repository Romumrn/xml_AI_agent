import requests
import json
import re
import csv
import time
from datetime import datetime
import sys


filename = sys.argv[1]

def ask_ollama(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.ok:
        return response.json()["response"]
    else:
        raise Exception(f"Ollama request failed for model {model}: {response.text}")

# === Load XML ===
with open(filename, "r", encoding="utf-8") as f:
    xml_data = f.read()

biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)

# === Config ===
models_to_test = [
    "deepseek-r1:8b", 
    "codestral:latest",
    "codellama:7b",
    "gemma3:4b",
    "qwen3:8b",
    "mistral:latest",
    
]
# === Prepare CSV ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"ollama_benchmark_{timestamp}.csv"

print( f"Number of blocks : {len(biosample_blocks)}")
with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "model", "block_index", "num_samples", "time_sec", "response"
    ])

    for model in models_to_test:
        print(f"\n--- Testing model: {model} ---")

        for i in range(0, len(biosample_blocks)):
            chunk = biosample_blocks[i]

#             message = f"""
# For each <BioSample> entry, extract the following fields:
# - Accession → from the `accession` attribute of <BioSample>, not the "id"
# - Latitude and Longitude → look for values in tags or attributes, or coordinates written as "lat;long" or similar (e.g., "41.40338;-2.17403")
# - Region → optional, from any field named "region", "state", "province", "area", etc.
# - Country → from geographic location fields or country-related tags

# Format requirement:
# Return ONLY a table in **CSV format**, with this exact header:
# accession,latitude,longitude,region,country

# If any field is missing, write "NA" (without quotes).

# Example output:
# accession,latitude,longitude,region,country  
# 534000000009,-90.0000,20.000000,Something,Japanos

# Process this XML input:
# <xml>
# {block_text}
# </xml>
# """
            message = f"""
Extract the most accurate geographic location information from this XML block (latitude and longitude or region or country for instance) 
Return ONLY the location string with no additional text. If the location is not possible write "NA"
XML Block:
{chunk}
            """
            
            start = time.time()
            response = ask_ollama(message, model=model)
            elapsed = time.time() - start

            response = response.split("</think>")[-1].strip()

            print(f"\n[Model: {model}] Block {i} processed in {elapsed:.2f} sec.")
            print(response.strip())

            writer.writerow([
                model,
                i,
                len(chunk),
                round(elapsed, 2),
                response.strip().replace("\n", " | ")
            ])

print(f"\n✅ Benchmark complete. Results saved to: {output_csv}")
