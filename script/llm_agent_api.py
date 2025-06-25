import requests
import re
import csv
import time
import argparse
import sys
import json
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LocationResult:
    """Classe pour structurer les résultats d'extraction de localisation"""
    accession: str
    regex_location: Optional[str] = None
    hapmap_location: Optional[str] = None
    llm_location: Optional[str] = None
    resolved_coordinates: Optional[str] = None
    final_location: Optional[str] = None
    processing_method: Optional[str] = None

class LocationExtractor:
    def __init__(self, model: str, coord_model: str, hapmap_file: str = "script/hapmap.json"):
        self.model = model
        self.coord_model = coord_model
        self.hapmap_population_info = self._load_hapmap_data(hapmap_file)
        
    def _load_hapmap_data(self, hapmap_file: str) -> Dict:
        """Charge les données HapMap de manière sécurisée"""
        try:
            with open(hapmap_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"HapMap file {hapmap_file} not found. HapMap resolution disabled.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {hapmap_file}")
            return {}

    def extract_coordinates_from_mixed_text(self, text: str) -> Optional[str]:
        """Extrait les coordonnées d'un texte mixte (coordonnées + noms de lieux)"""
        if not text:
            return None
            
        # Patterns pour différents formats de coordonnées
        coord_patterns = [
            # Format décimal simple : lat,lon ou lat, lon
            r'(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',
            # Format avec directions : 12.34 N, 56.78 W
            r'(\d+\.?\d*)\s*([NS])\s*,?\s*(\d+\.?\d*)\s*([EW])',
            # Format DMS : 12°34'56"N 78°90'12"W
            r'(\d+)°(\d+)\'(\d+\.?\d*)"?\s*([NS])\s*,?\s*(\d+)°(\d+)\'(\d+\.?\d*)"?\s*([EW])'
        ]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            
            # Test du format décimal simple
            match = re.search(coord_patterns[0], line)
            if match:
                lat, lon = match.groups()
                try:
                    lat_val, lon_val = float(lat), float(lon)
                    if -90 <= lat_val <= 90 and -180 <= lon_val <= 180:
                        return f"{lat_val},{lon_val}"
                except ValueError:
                    continue
            
            # Test du format avec directions
            match = re.search(coord_patterns[1], line)
            if match:
                lat_val, lat_dir, lon_val, lon_dir = match.groups()
                try:
                    lat_num = float(lat_val) * (-1 if lat_dir.upper() == 'S' else 1)
                    lon_num = float(lon_val) * (-1 if lon_dir.upper() == 'W' else 1)
                    if -90 <= lat_num <= 90 and -180 <= lon_num <= 180:
                        return f"{lat_num},{lon_num}"
                except ValueError:
                    continue
        
        return None

    def is_valid_coordinate(self, coord: str) -> bool:
        """Version améliorée de la validation des coordonnées"""
        if not coord or not isinstance(coord, str):
            return False
        
        coord = coord.strip().lower()
        
        # Termes invalides
        invalid_terms = [
            "not determined", "not collected", "missing", "na", "n/a", 
            "unknown", "not applicable", "not provided", "none", "null"
        ]
        
        if any(term in coord for term in invalid_terms):
            return False
        
        # Vérification basique de la présence de chiffres
        if not re.search(r'\d', coord):
            return False
        
        # Test de validation des valeurs numériques pour le format décimal
        if re.match(r'^-?\d+\.?\d*,\s*-?\d+\.?\d*$', coord):
            try:
                lat_str, lon_str = coord.split(',')
                lat, lon = float(lat_str.strip()), float(lon_str.strip())
                return -90 <= lat <= 90 and -180 <= lon <= 180
            except (ValueError, IndexError):
                return False
        
        # Autres patterns de coordonnées valides
        coord_patterns = [
            r'^-?\d+°\d+\'\d*\.?\d*"?\s*[NS],?\s*-?\d+°\d+\'\d*\.?\d*"?\s*[EW]$',  # DMS
            r'^-?\d+\.\d+\s*[NS],?\s*-?\d+\.\d+\s*[EW]$',  # Décimal avec direction
        ]
        
        return any(re.match(pattern, coord, re.IGNORECASE) for pattern in coord_patterns)

    def is_pure_coordinates(self, text: str) -> bool:
        """Vérifie si le texte contient uniquement des coordonnées (pas de noms de lieux)"""
        if not text:
            return False
            
        # Nettoie le texte
        cleaned = text.strip()
        
        # Vérifie s'il s'agit de coordonnées simples
        simple_coord_pattern = r'^-?\d+\.?\d*\s*,\s*-?\d+\.?\d*$'
        if re.match(simple_coord_pattern, cleaned):
            return True
            
        # Vérifie s'il s'agit de coordonnées avec directions
        dir_coord_pattern = r'^-?\d+\.?\d*\s*[NS]\s*,?\s*-?\d+\.?\d*\s*[EW]$'
        if re.match(dir_coord_pattern, cleaned, re.IGNORECASE):
            return True
            
        return False

    def extract_place_names_from_mixed_text(self, text: str) -> Optional[str]:
        """Extrait les noms de lieux d'un texte mixte, en excluant les coordonnées"""
        if not text:
            return None
            
        lines = text.split('\n')
        place_names = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip les lignes qui contiennent des coordonnées
            if self.is_pure_coordinates(line):
                continue
                
            # Skip les lignes qui contiennent principalement des chiffres
            if re.match(r'^[\d\s\.,°\'"NSEW-]+$', line):
                continue
                
            # Garde les lignes qui ressemblent à des noms de lieux
            if re.search(r'[a-zA-Z]', line) and len(line) > 2:
                place_names.append(line)
        
        return ', '.join(place_names) if place_names else None

    def ask_ollama_with_tools(self, prompt: str, model: str, tools: list, max_retries: int = 3) -> Dict[str, Any]:
        """Version améliorée avec meilleure gestion d'erreurs"""
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
            "tools": tools
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ConnectionError:
                logger.error(f"Cannot connect to Ollama (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                logger.error(f"Ollama timeout (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Ollama request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
        logger.error("All Ollama attempts failed")
        return {}

    def geocode_tool(self, place_name: str) -> Optional[str]:
        """Version améliorée qui évite le géocodage de coordonnées valides"""
        if not place_name or place_name.lower() in ['na', 'n/a', 'unknown']:
            return None
            
        # CRITIQUE: Vérifier si c'est déjà des coordonnées valides
        if self.is_valid_coordinate(place_name):
            logger.debug(f"Skipping geocoding of valid coordinates: {place_name}")
            return place_name
            
        # Si c'est un texte mixte, extraire les coordonnées d'abord
        extracted_coords = self.extract_coordinates_from_mixed_text(place_name)
        if extracted_coords:
            logger.debug(f"Extracted coordinates from mixed text '{place_name}': {extracted_coords}")
            return extracted_coords
            
        # Extraire seulement les noms de lieux pour le géocodage
        place_names_only = self.extract_place_names_from_mixed_text(place_name)
        if not place_names_only:
            logger.debug(f"No place names found in: {place_name}")
            return None
            
        logger.debug(f"Geocoding place names: {place_names_only}")
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": place_names_only, "format": "json", "limit": 1}
        headers = {"User-Agent": "geo-agent/1.0 (research purposes)"}
        
        try:
            # Rate limiting pour respecter les limites de Nominatim
            time.sleep(1)
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                
                # Validation des coordonnées
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coord = f"{lat},{lon}"
                    logger.debug(f"Geocoded '{place_names_only}' to {coord}")
                    return coord
                    
        except requests.exceptions.RequestException as e:
            logger.warning(f"Geocoding failed for '{place_names_only}': {e}")
        except (ValueError, KeyError, IndexError) as e:
            logger.warning(f"Invalid geocoding response for '{place_names_only}': {e}")
            
        return None

    def extract_coordinates_regex(self, block: str) -> Optional[str]:
        """Extraction par regex avec patterns améliorés"""
        patterns = [
            # Coordonnées combinées
            (r'<Attribute\s+[^>]*?attribute_name="lat_lon"[^>]*>([^<]+)</Attribute>', None),
            # Latitude/longitude séparées (plusieurs variantes)
            (r'<Attribute\s+[^>]*?attribute_name="geographic location \(latitude\)"[^>]*>([^<]+)</Attribute>',
             r'<Attribute\s+[^>]*?attribute_name="geographic location \(longitude\)"[^>]*>([^<]+)</Attribute>'),
            (r'<Attribute\s+[^>]*?attribute_name="latitude"[^>]*>([^<]+)</Attribute>',
             r'<Attribute\s+[^>]*?attribute_name="longitude"[^>]*>([^<]+)</Attribute>'),
            # Patterns avec unités
            (r'<Attribute\s+[^>]*?attribute_name="latitude"[^>]*?unit="DecimalDegrees"[^>]*>([^<]+)</Attribute>',
             r'<Attribute\s+[^>]*?attribute_name="longitude"[^>]*?unit="DecimalDegrees"[^>]*>([^<]+)</Attribute>'),
        ]

        for lat_pattern, lon_pattern in patterns:
            lat_match = re.search(lat_pattern, block, re.IGNORECASE)
            
            if lat_match:
                if lon_pattern is None:
                    # Coordonnées combinées
                    coords_text = lat_match.group(1).strip()
                    
                    # Essayer d'extraire les coordonnées du texte mixte
                    extracted_coords = self.extract_coordinates_from_mixed_text(coords_text)
                    if extracted_coords:
                        return extracted_coords
                        
                    # Si c'est déjà des coordonnées valides
                    if self.is_valid_coordinate(coords_text):
                        return self._format_coordinate(coords_text)
                else:
                    # Coordonnées séparées
                    lon_match = re.search(lon_pattern, block, re.IGNORECASE)
                    if lon_match:
                        lat, lon = lat_match.group(1).strip(), lon_match.group(1).strip()
                        coords = f"{lat},{lon}"
                        
                        if self.is_valid_coordinate(coords):
                            return self._format_coordinate(coords)
        
        return None

    def _format_coordinate(self, coord: str) -> str:
        """Standardise le format des coordonnées"""
        if not coord:
            return coord
        
        coord = re.sub(r'\s+', ' ', coord.strip())
        
        # Gestion des coordonnées avec directions (ex: "21.434025 N 157.787827 W")
        dms_pattern = r'^(-?\d+\.?\d*)\s+([NS])\s+(-?\d+\.?\d*)\s+([EW])$'
        match = re.match(dms_pattern, coord, re.IGNORECASE)
        if match:
            lat, lat_dir, lon, lon_dir = match.groups()
            lat_val = float(lat) * (-1 if lat_dir.upper() == 'S' else 1)
            lon_val = float(lon) * (-1 if lon_dir.upper() == 'W' else 1)
            return f"{lat_val},{lon_val}"
        
        return coord

    def extract_hapmap_location(self, block: str) -> Optional[str]:
        """Extraction améliorée des codes de population HapMap"""
        if not self.hapmap_population_info:
            return None
            
        pop_patterns = [
            r'<Attribute\s+[^>]*?attribute_name="population"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="population description"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="ethnicity"[^>]*>([^<]+)</Attribute>',
        ]
        
        for pattern in pop_patterns:
            pop_match = re.search(pattern, block, re.IGNORECASE)
            if pop_match:
                pop_code = pop_match.group(1).strip().upper()
                
                # Recherche dans les données HapMap
                for dataset in self.hapmap_population_info.values():
                    for code, info in dataset.items():
                        if pop_code == code or code in pop_code or pop_code in code:
                            if isinstance(info, dict) and "coordinate" in info:
                                coords = info["coordinate"]
                                return f"{coords[0]},{coords[1]}"
                            
        return None

    def llm_extract_location(self, block: str) -> Optional[str]:
        """Version améliorée de l'extraction LLM avec prompt optimisé"""
        prompt = f"""
You are a specialized geographic data extraction assistant. Analyze this XML block and extract ONLY geographic location information.

EXTRACT:
✓ City, region, country names
✓ Latitude/longitude coordinates (any format)
✓ Geographic landmarks or areas

IGNORE:
✗ Species names, organism information
✗ Sample types, body sites
✗ Project names, study information
✗ Technical metadata

OUTPUT RULES:
- Return ONLY the most specific geographic identifier found
- For coordinates: use decimal format (e.g., "40.7128,-74.0060")
- For place names: use the most specific available (e.g., "New York, USA" not just "USA")
- If no geographic information exists, return exactly: "NA"

XML BLOCK:
{block}

GEOGRAPHIC LOCATION:"""

        try:
            response = self.ask_ollama_with_tools(prompt, self.model, [], max_retries=2)
            if response and "response" in response:
                location = self._clean_llm_response(response["response"])
                return location if location and location.upper() != "NA" else None
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            
        return None

    def _clean_llm_response(self, response: str) -> Optional[str]:
        """Nettoyage amélioré des réponses LLM"""
        if not response:
            return None
            
        # Suppression des balises de réflexion
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()
        
        # Suppression des préfixes courants
        prefixes = ["location:", "geographic location:", "answer:", "result:"]
        for prefix in prefixes:
            if response.lower().startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Vérification des termes invalides
        invalid_terms = ["n/a", "na", "none", "missing", "not determined", "unknown"]
        if any(term in response.lower() for term in invalid_terms):
            return None
            
        return response if response else None

    def resolve_place_to_coordinates(self, place_name: str) -> Optional[str]:
        """Version améliorée de la résolution de lieu en coordonnées"""
        if not place_name or place_name.lower() in ['na', 'n/a']:
            return None

        # CRITIQUE: Vérifier d'abord si c'est déjà des coordonnées
        if self.is_valid_coordinate(place_name):
            logger.debug(f"Input is already valid coordinates: {place_name}")
            return place_name

        # Extraire les coordonnées d'un texte mixte
        extracted_coords = self.extract_coordinates_from_mixed_text(place_name)
        if extracted_coords:
            logger.debug(f"Extracted coordinates from mixed input: {extracted_coords}")
            return extracted_coords

        # Tentative directe avec l'API de géocodage (seulement pour les noms de lieux)
        direct_coords = self.geocode_tool(place_name)
        if direct_coords:
            return direct_coords

        # Si échec, essayer avec LLM + outil
        llm_geocode_tool = {
            "name": "geocode_tool",
            "description": "Fetch decimal latitude,longitude for a place name",
            "parameters": {
                "type": "object",
                "properties": {
                    "place_name": {
                        "type": "string",
                        "description": "A place to geocode"
                    }
                },
                "required": ["place_name"]
            }
        }

        prompt = f"""
You are a geocoding assistant. For the given place name, either:
1. If you know the approximate coordinates, output: COORDS:<lat>,<lon>
2. Otherwise, call the geocoding tool: TOOL_CALL geocode_tool(place_name="{place_name}")

IMPORTANT: If the input already contains coordinates, extract and return them as COORDS:<lat>,<lon>

PLACE: {place_name}
"""

        try:
            resp = self.ask_ollama_with_tools(prompt, self.coord_model, tools=[llm_geocode_tool])
            
            # Vérification des appels d'outils
            if resp.get("tool_calls"):
                params = resp["tool_calls"][0]["parameters"]
                coords = self.geocode_tool(params["place_name"])
                return coords if coords and self.is_valid_coordinate(coords) else None

            # Vérification des réponses directes
            raw = resp.get("response", "")
            if raw.upper().startswith("COORDS:"):
                candidate = raw.split("COORDS:", 1)[1].strip()
                return candidate if self.is_valid_coordinate(candidate) else None
                
        except Exception as e:
            logger.error(f"Coordinate resolution failed for '{place_name}': {e}")

        return None

    def process_biosample_block(self, block: str, accession: str) -> LocationResult:
        """Traitement complet d'un bloc BioSample"""
        result = LocationResult(accession=accession)
        
        # Étape 1: Extraction par regex
        result.regex_location = self.extract_coordinates_regex(block)
        if result.regex_location:
            result.final_location = result.regex_location
            result.processing_method = "regex"
            logger.debug(f"{accession}: Found coordinates via regex: {result.regex_location}")
            return result

        # Étape 2: Tentative HapMap
        result.hapmap_location = self.extract_hapmap_location(block)
        if result.hapmap_location:
            result.final_location = result.hapmap_location
            result.processing_method = "hapmap"
            logger.debug(f"{accession}: Found coordinates via HapMap: {result.hapmap_location}")
            return result

        # Étape 3: Extraction LLM
        result.llm_location = self.llm_extract_location(block)
        if result.llm_location:
            # Tentative de résolution en coordonnées
            if self.is_valid_coordinate(result.llm_location):
                result.final_location = result.llm_location
                result.processing_method = "llm_direct"
                logger.debug(f"{accession}: LLM returned direct coordinates: {result.llm_location}")
            else:
                # Résolution de lieu en coordonnées
                result.resolved_coordinates = self.resolve_place_to_coordinates(result.llm_location)
                if result.resolved_coordinates:
                    result.final_location = result.resolved_coordinates
                    result.processing_method = "llm_resolved"
                    logger.debug(f"{accession}: Resolved '{result.llm_location}' to {result.resolved_coordinates}")
                else:
                    # Garder le nom de lieu comme résultat final
                    result.final_location = result.llm_location
                    result.processing_method = "llm_place"
                    logger.debug(f"{accession}: Kept place name: {result.llm_location}")

        return result

def main():
    parser = argparse.ArgumentParser(description="Extract location data from SRA XML with improved coordinate handling.")
    parser.add_argument("file", help="Input XML file path")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name for general processing")
    parser.add_argument("--coord_model", default="llama3.1", help="Ollama model for coordinate resolution")
    parser.add_argument("--output", default="output.csv", help="Output CSV file path")
    parser.add_argument("--hapmap", default="script/hapmap.json", help="HapMap population data file")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for progress reporting")
    
    args = parser.parse_args()

    # Vérification de l'existence du fichier
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            xml_data = f.read()
    except FileNotFoundError:
        logger.error(f"Input file {args.file} not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        sys.exit(1)

    # Initialisation de l'extracteur
    extractor = LocationExtractor(args.model, args.coord_model, args.hapmap)
    
    # Extraction des blocs BioSample
    biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    logger.info(f"Found {len(biosample_blocks)} BioSample blocks in {args.file}")

    if not biosample_blocks:
        logger.warning("No BioSample blocks found in the input file")
        sys.exit(1)

    # Traitement et écriture des résultats
    try:
        with open(args.output, "w", newline='', encoding='utf-8') as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "Accession", "Final_Location", "Processing_Method", 
                "Regex_Location", "HapMap_Location", "LLM_Location", "Resolved_Coordinates"
            ])

            start_time = time.time()
            processed = 0
            successful_extractions = 0

            for i, block in enumerate(biosample_blocks, 1):
                # Extraction de l'accession
                accession_match = re.search(r'accession="([^"]+)"', block)
                accession = accession_match.group(1) if accession_match else f"UNKNOWN_{i}"

                # Traitement du bloc
                try:
                    result = extractor.process_biosample_block(block, accession)
                    
                    # Écriture du résultat
                    writer.writerow([
                        result.accession,
                        result.final_location or "",
                        result.processing_method or "",
                        result.regex_location or "",
                        result.hapmap_location or "",
                        result.llm_location or "",
                        result.resolved_coordinates or ""
                    ])
                    
                    if result.final_location:
                        successful_extractions += 1
                        
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {accession}: {e}")
                    # Écriture d'une ligne d'erreur
                    writer.writerow([accession, "", "error", "", "", "", ""])
                    processed += 1

                # Rapport de progression
                if i % args.batch_size == 0 or i == len(biosample_blocks):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed if processed > 0 else 0
                    success_rate = (successful_extractions / processed * 100) if processed > 0 else 0
                    
                    logger.info(f"Progress: {i}/{len(biosample_blocks)} blocks processed "
                              f"({success_rate:.1f}% success rate, {avg_time:.2f}s/block)")

            # Statistiques finales
            total_time = time.time() - start_time
            logger.info(f"Processing complete! {successful_extractions}/{processed} successful extractions "
                       f"({successful_extractions/processed*100:.1f}% success rate)")
            logger.info(f"Total processing time: {total_time:.2f}s, Average: {total_time/processed:.2f}s/block")
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()