import requests
import json
import re
import csv
import time
import argparse
import sys
import json

def ask_ollama(prompt, model, max_retries=3):
    """Send a prompt to the Ollama API and return the response text."""
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
                print("[ERROR] All retries failed. Returning empty string.")
                return ""

def is_valid_coordinate(coord):
    """Check if the coordinate string appears to contain valid location data."""
    if not coord or not isinstance(coord, str):
        return False
    
    coord = coord.strip().lower()
    
    # Skip common non-coordinate indicators
    invalid_terms = [
        "not determined", 
        "not collected",
        "missing",
        "na",
        "n/a",
        "unknown",
        "not applicable",
        "not provided"
    ]
    
    if any(term in coord for term in invalid_terms):
        return False
    
    # Check for at least one digit and valid coordinate characters
    if not re.search(r'\d', coord):
        return False
    
    # Check for valid coordinate patterns
    coord_patterns = [
        r'^-?\d+\.\d+,\s*-?\d+\.\d+$',  # Decimal degrees with comma
        r'^-?\d+°\d+\'\d+\.?\d*\"\s*[NS],\s*-?\d+°\d+\'\d+\.?\d*\"\s*[EW]$',  # DMS
        r'^-?\d+\.\d+\s*[NS],\s*-?\d+\.\d+\s*[EW]$',  # Decimal with direction
        r'^-?\d+\.\d+\s+[NS]\s+-?\d+\.\d+\s+[EW]$',  # Space-separated with directions
        r'^-?\d+\.\d+\s+[NS]\s*-?\d+\.\d+\s+[EW]$',   # Space-separated with directions, optional space
        r'^-?\d+\.\d+\s*[NS]\s+-?\d+\.\d+\s*[EW]$'    # Space-separated with directions, flexible spacing
    ]
    
    return any(re.match(pattern, coord, re.IGNORECASE) for pattern in coord_patterns)

def format_coordinate(coord):
    """Standardize coordinate format for consistent output"""
    if not coord:
        return coord
    
    # Remove extra spaces and normalize formatting
    coord = re.sub(r'\s+', ' ', coord.strip())
    
    # Handle space-separated coordinates with directions (like "21.434025 N 157.787827 W")
    if re.match(r'^-?\d+\.\d+\s+[NS]\s+-?\d+\.\d+\s+[EW]$', coord, re.IGNORECASE):
        parts = coord.split()
        return f"{parts[0]} {parts[1]}, {parts[2]} {parts[3]}"
    
    return coord

def extract_coordinates(block):
    """Extract and validate coordinates from XML block using different patterns."""
    patterns = [
        # lat_lon combined (space separated with directions)
        (
            r'<Attribute\s+[^>]*?attribute_name="lat_lon"[^>]*>([^<]+)</Attribute>',
            None,
            lambda latlon, _: format_coordinate(latlon.strip())
        ),
        # Separate latitude/longitude
        (
            r'<Attribute\s+[^>]*?attribute_name="geographic location \(latitude\)"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="geographic location \(longitude\)"[^>]*>([^<]+)</Attribute>',
            lambda lat, lon: format_coordinate(f"{lat.strip()},{lon.strip()}")
        ),
        # Decimal latitude/longitude
        (
            r'<Attribute\s+[^>]*?attribute_name="latitude"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="longitude"[^>]*>([^<]+)</Attribute>',
            lambda lat, lon: format_coordinate(f"{lat.strip()},{lon.strip()}")
        )
    ]

    for lat_pattern, lon_pattern, formatter in patterns:
        lat_match = re.search(lat_pattern, block) if lat_pattern else None
        lon_match = re.search(lon_pattern, block) if lon_pattern else None
        
        if lat_match and (lon_pattern is None or lon_match):
            lat = lat_match.group(1)
            lon = lon_match.group(1) if lon_match else None
            coords = formatter(lat, lon) if lon else formatter(lat, None)
            
            if is_valid_coordinate(coords):
                return coords
    
    return None

def extract_coordinates(block):
    """Extract and validate coordinates from XML block using different patterns."""
    patterns = [
        # lat_lon combined (space separated with directions)
        (
            r'<Attribute\s+[^>]*?attribute_name="lat_lon"[^>]*>([^<]+)</Attribute>',
            None,
            lambda latlon, _: format_coordinate(latlon.strip())
        ),
        # Separate latitude/longitude
        (
            r'<Attribute\s+[^>]*?attribute_name="geographic location \(latitude\)"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="geographic location \(longitude\)"[^>]*>([^<]+)</Attribute>',
            lambda lat, lon: format_coordinate(f"{lat.strip()},{lon.strip()}")
        ),
        # Decimal latitude/longitude
        (
            r'<Attribute\s+[^>]*?attribute_name="latitude"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="longitude"[^>]*>([^<]+)</Attribute>',
            lambda lat, lon: format_coordinate(f"{lat.strip()},{lon.strip()}")
        ),
        # New pattern: DecimalDegrees unit
        (
            r'<Attribute\s+[^>]*?attribute_name="latitude"[^>]*?unit="DecimalDegrees"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="longitude"[^>]*?unit="DecimalDegrees"[^>]*>([^<]+)</Attribute>',
            lambda lat, lon: format_coordinate(f"{lat.strip()},{lon.strip()}")
        )
    ]

    for lat_pattern, lon_pattern, formatter in patterns:
        lat_match = re.search(lat_pattern, block) if lat_pattern else None
        lon_match = re.search(lon_pattern, block) if lon_pattern else None
        
        if lat_match and (lon_pattern is None or lon_match):
            lat = lat_match.group(1)
            lon = lon_match.group(1) if lon_match else None
            coords = formatter(lat, lon) if lon else formatter(lat, None)
            
            if is_valid_coordinate(coords):
                return coords
    
    return None

def process_xml_file(filename, model, hapmap_population_info, output_file="output.csv"):
    """Process XML file and extract location data with validation."""
    with open(filename, "r", encoding="utf-8") as f:
        xml_data = f.read()

    biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    print(f"[INFO] Found {len(biosample_blocks)} BioSample blocks in {filename}")

    with open(output_file, "w", newline='', encoding='utf-8') as output:
        writer = csv.writer(output)
        writer.writerow(["Accession", "Location", "LLM_loc_1", "LLM_loc_2"])

        for i, block in enumerate(biosample_blocks, 1):
            start_time = time.time()
            
            if i % 100 == 0 or i == 1:
                print(f"[INFO] Processing block {i}/{len(biosample_blocks)}")

            # Extract accession
            accession_match = re.search(r'accession="([^"]+)"', block)
            accession = accession_match.group(1) if accession_match else f"UNKNOWN_{i}"

            # Initialize output variables
            python_location = ""
            llm_location_1 = ""
            llm_location_2 = ""

            # Try to extract coordinates with Python
            python_location = extract_coordinates(block)
            if python_location:
                print(f"[INFO] {accession}: Found valid coordinates: {python_location}")

            hapmap_location = extract_hapmap_location(block, hapmap_population_info)

            if hapmap_location:
                python_location = hapmap_location
                print(f"[INFO] {accession}: Found HapMap location: {hapmap_location}")
    
            # If Python found nothing, try with LLM prompts
            if not python_location:
                # First prompt
                prompt1 = f"""Extract the geographic location from this XML block. 
If no location is provided, answer "NA". Return only the location string or empty if none found.
XML Block: {block}"""

                response1 = ask_ollama(prompt1, model).strip()
                llm_location_1 = clean_llm_response(response1)

                # Second prompt
                prompt2 = f"""
You are a precise data extraction assistant. Your task is to:
1️ Analyze the following XML block carefully.
2️ Extract the most specific *geographic location* or *valid coordinates* provided within the XML (from any attribute, tag, or value).
3️Prioritize true geographic identifiers:
    - Place names (e.g., city, region, landmark, country)
    - Valid latitude/longitude coordinates (in decimal degrees or degrees-minutes-seconds)
4️ Ignore unrelated information, including:
    - Species names
    - Host organisms
    - Body sites
    - Project or study information
5️ When multiple geographic indicators exist (e.g., geo_loc_name, location, lat_lon, population description):
    - Select the most specific and relevant for geographic positioning (e.g., city over country, coordinates over place name).

📌 Output format:
- Return **only** the location string or coordinates (no additional text, explanation, or formatting).
- If no valid location or coordinates are found, return exactly: `NA`

Here is the XML block:
{block}
"""


                response2 = ask_ollama(prompt2, model).strip()
                llm_location_2 = clean_llm_response(response2)

                if llm_location_1:
                    print(f"[INFO] {accession}: LLM prompt 1 -> {llm_location_1}")
                if llm_location_2:
                    print(f"[INFO] {accession}: LLM prompt 2 -> {llm_location_2}")

            writer.writerow([accession, python_location or "", llm_location_1 or "", llm_location_2 or ""])
            
            elapsed = time.time() - start_time
            if i % 100 == 0:
                print(f"[INFO] Average time per block so far: {elapsed / i:.2f}s")

    print(f"[INFO] Processing complete! Results saved to {output_file}")


def clean_llm_response(response):
    """Helper to clean LLM response text."""
    response = response.split("</think>")[-1].strip() if "</think>" in response else response
    response = re.sub(r'^[^A-Za-z0-9\-\.°\'\"\s]*|[^A-Za-z0-9\-\.°\'\"\s]*$', '', response)
    if any(term in response.lower() for term in ["n/a", "na", "none", "missing", "not determined"]):
        return "NA"
    return response

def extract_hapmap_location(block, hapmap_population_info):
    """Try to extract HapMap population code and map to location."""
    # Look for population code in common tags
    pop_match = re.search(
        r'<Attribute\s+[^>]*?attribute_name="(?:population|population description)"[^>]*>([^<]+)</Attribute>',
        block, re.IGNORECASE)
    
    if pop_match:
        pop_code = pop_match.group(1).strip().upper()
        # Check if the code is in our mapping (exact match or part of description)
        for code, info in hapmap_population_info.items():
            if pop_code == code or code in pop_code:
                return info["location"]
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Extract location data from SRA XML.")
    parser.add_argument("file", help="Input XML file path")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--output", default="output.csv", help="Output CSV file path")

    args = parser.parse_args()

    with open("script/hapmap.json", "r", encoding="utf-8") as f:
        hapmap_population_info = json.load(f)
    
    print(f"[INFO] Starting processing with model: {args.model}")
    process_xml_file(args.file, args.model, hapmap_population_info, args.output)


if __name__ == "__main__":
    main()