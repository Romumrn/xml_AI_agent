import requests
import json
import re
import csv
import time
import argparse
import sys

def ask_ollama(prompt, model, max_retries=3):
    """
    Send a prompt to the Ollama API and return the response text.
    Implements retries with exponential backoff in case of failure.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt, 
        "stream": False,
        "options": {"temperature": 0.0}
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            return response.json()["response"]
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"[ERROR] Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print("[ERROR] All retries failed. Returning 'N/A'.")
                return "N/A"

def process_xml_file(filename, model, output_file="output.csv"):
    """
    Process an XML file, extract location data, and write results to a CSV.
    Prints the mean processing time per block.
    """
    with open(filename, "r", encoding="utf-8") as f:
        xml_data = f.read()

    biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    print(f"[INFO] Found {len(biosample_blocks)} BioSample blocks in {filename}")

    total_time = 0

    with open(output_file, "w", newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(["Accession", "Location"])

        for i, block in enumerate(biosample_blocks):
            start_time = time.time()

            if (i + 1) % 100 == 0 or i == 0:
                print(f"[INFO] Processing block {i + 1}/{len(biosample_blocks)}")

            accession_match = re.search(r'accession="([^"]+)"', block)
            accession = accession_match.group(1) if accession_match else "UNKNOWN_ACCESSION"

            lat_match = re.search(
                r'<Attribute\s+[^>]*?attribute_name="geographic location \(latitude\)"[^>]*>([^<]+)</Attribute>',
                block
            )
            lon_match = re.search(
                r'<Attribute\s+[^>]*?attribute_name="geographic location \(longitude\)"[^>]*>([^<]+)</Attribute>',
                block
            )

            if lat_match and lon_match:
                location = f"{lat_match.group(1)},{lon_match.group(1)}"
                print(f"[INFO] {accession}: {location} (direct coordinates)")
            else:
                prompt = f"""
                Extract the most accurate geographic location information from this XML block (latitude and longitude or region or country for instance). 
                Return ONLY the location string with no additional text. If the location is not possible write "NA".
                XML Block:
                {block}
                """
                response = ask_ollama(prompt, model).strip()
                response = response.split("</think>")[-1].strip()
                location = re.sub(r'^[^A-Za-z0-9]*|[^A-Za-z0-9]*$', '', response)
                location = location if location else "N/A"
                print(f"[INFO] {accession}: {location} (LLM extracted)")

            writer.writerow([accession, location])

            end_time = time.time()
            elapsed = end_time - start_time
            total_time += elapsed
            mean_time = total_time / (i + 1)
            print(f"[INFO] Mean processing time per block: {mean_time:.2f} seconds")

    print(f"[INFO] Processing complete! Output saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract location data from SRA XML using a local LLM via Ollama."
    )
    parser.add_argument("file", help="Path to the input XML file")
    parser.add_argument("--model", default="qwen3:4b", help="LLM model name (default: qwen3:4b)")
    parser.add_argument("--output", default="output.csv", help="Output CSV file (default: output.csv)")

    args = parser.parse_args()

    print(f"[INFO] Using model: {args.model}")
    print(f"[INFO] Input file: {args.file}")
    print(f"[INFO] Output file: {args.output}")

    process_xml_file(args.file, args.model, args.output)

if __name__ == "__main__":
    main()
