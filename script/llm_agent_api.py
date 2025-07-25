import requests
import re
import csv
import time
import argparse
import sys
import json
import logging
from typing import Optional, Dict, Any, Tuple
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from tqdm import tqdm


# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LocationResult:
    """Classe pour structurer les résultats d'extraction de localisation"""
    accession: str
    direct_location: Optional[str] = None  # Fusion de regex et hapmap
    direct_method: Optional[str] = None    # "regex" ou "hapmap"
    llm_location: Optional[str] = None
    resolved_coordinates: Optional[str] = None
    final_location: Optional[str] = None
    processing_method: Optional[str] = None
    source_attribute: Optional[str] = None  # Nouvel attribut pour la source XML

class LocationExtractor:
    def __init__(self, model: str, coord_model: str, attr_model: str, hapmap_file: str = "script/hapmap.json"):
        self.model = model
        self.coord_model = coord_model
        self.attr_model = attr_model  # Nouveau modèle pour l'extraction d'attributs
        self.hapmap_population_info = self._load_hapmap_data(hapmap_file)
        # Initialisation du logger CSV
        self.llm_logger = self._init_llm_logger()
        
    def _init_llm_logger(self):
        """Initialise le logger CSV pour les réponses LLM"""
        log_file = "llm_responses.csv"
        try:
            # Ouvrir le fichier en mode append
            f = open(log_file, 'a', newline='', encoding='utf-8')
            writer = csv.writer(f)
            # Écrire l'en-tête si le fichier est vide
            if f.tell() == 0:
                writer.writerow(["Accession", "Model", "Timestamp", "Response"])
            return {"file": f, "writer": writer}
        except Exception as e:
            logger.error(f"Failed to initialize LLM logger: {e}")
            return None

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
    def ask_ollama_with_tools(self, prompt: str, model: str, tools: list, accession: str, max_retries: int = 3) -> Dict[str, Any]:
        """Version améliorée avec meilleure gestion d'erreurs et logging"""
        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0},
            "tools": tools
        }
        
        response_data = {}
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=100)
                response.raise_for_status()
                response_data = response.json()
                break
            except requests.exceptions.ConnectionError as e:
                logger.error(f"Cannot connect to Ollama (attempt {attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.Timeout as e:
                logger.error(f"Ollama timeout (attempt {attempt+1}/{max_retries}) {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"{accession} Ollama request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        else:
            logger.error(f"{accession} All Ollama attempts failed")
            response_data = {"error": "All attempts failed"}
        
        # Logguer la réponse dans le fichier CSV
        
        self._log_llm_response(accession, model, response_data)
        return response_data

    def llm_extract_location(self, block: str, accession: str) -> Optional[str]:
        """Extraction de localisation avec LLM"""
        prompt = f"""
Extract ONLY geographic location information from this XML block:

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
- For place names: use specific format (e.g., "New York, USA")
- If no geographic information: return "NA"

XML BLOCK:
{block}

GEOGRAPHIC LOCATION:"""

        try:
            response = self.ask_ollama_with_tools(prompt, self.model, [], accession, max_retries=2)
            if response and "response" in response:
                location = self._clean_llm_response(response["response"])
                return location if location and location.upper() != "NA" else None
        except Exception as e:
            logger.error(f"LLM extraction failed for {accession}: {e}")
        return None
    
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

    def extract_coordinates_regex(self, block: str) -> Tuple[Optional[str], Optional[str]]:
        """Extraction par regex avec patterns améliorés - retourne (coordonnées, attribut_source)"""
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
                # Capturer l'attribut source complet
                source_attr = lat_match.group(0)
                
                if lon_pattern is None:
                    # Coordonnées combinées
                    coords_text = lat_match.group(1).strip()
                    
                    # Essayer d'extraire les coordonnées du texte mixte
                    extracted_coords = self.extract_coordinates_from_mixed_text(coords_text)
                    if extracted_coords:
                        return extracted_coords, source_attr
                        
                    # Si c'est déjà des coordonnées valides
                    if self.is_valid_coordinate(coords_text):
                        return self._format_coordinate(coords_text), source_attr
                else:
                    # Coordonnées séparées
                    lon_match = re.search(lon_pattern, block, re.IGNORECASE)
                    if lon_match:
                        lat, lon = lat_match.group(1).strip(), lon_match.group(1).strip()
                        coords = f"{lat},{lon}"
                        
                        if self.is_valid_coordinate(coords):
                            # Combiner les deux attributs sources
                            source_attr = f"{lat_match.group(0)} + {lon_match.group(0)}"
                            return self._format_coordinate(coords), source_attr
        
        return None, None

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

    def _log_llm_response(self, accession: str, model: str, response: dict):
        """Journalise les réponses LLM dans un fichier CSV (méthode interne)"""
        if not self.llm_logger:
            return
            
        try:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            response_text = response.get("response", str(response))  # Capture aussi les erreurs
            cleaned_response = response_text.replace("\n", "\\n").replace("\r", "").replace("\t", " ")
            
            self.llm_logger["writer"].writerow([
                accession,
                model,
                timestamp,
                cleaned_response[:10000]  # Limite la taille pour éviter les débordements
            ])
            self.llm_logger["file"].flush()
        except Exception as e:
            logger.error(f"Failed to log LLM response for {accession}: {e}")

    def extract_hapmap_location(self, block: str) -> Tuple[Optional[str], Optional[str]]:
        """Extraction améliorée des codes de population HapMap - retourne (coordonnées, attribut_source)"""
        if not self.hapmap_population_info:
            return None, None
            
        pop_patterns = [
            r'<Attribute\s+[^>]*?attribute_name="population"[^>]*>([^<]+)</Attribute>',
            r'<Attribute\s+[^>]*?attribute_name="population description"[^>]*>([^<]+)</Attribute>',
        ]
        
        for pattern in pop_patterns:
            pop_match = re.search(pattern, block, re.IGNORECASE)
            if pop_match:
                pop_code = pop_match.group(1).strip().upper()
                source_attr = pop_match.group(0)
                
                # Recherche dans les données HapMap
                for dataset in self.hapmap_population_info.values():
                    for code, info in dataset.items():
                        if pop_code == code or code in pop_code or pop_code in code:
                            if isinstance(info, dict) and "coordinate" in info:
                                coords = info["coordinate"]
                                return f"{coords[0]},{coords[1]}", source_attr
                            
        return None, None
    
    def extract_source_attribute(self, block: str, location_value: str, accession: str) -> Optional[str]:
        """Utilise un LLM pour extraire l'attribut source d'une valeur de localisation"""
        if not location_value:
            return None
            
        prompt = f"""
You are an XML attribute extraction specialist. Given an XML block and a location value, find the EXACT XML attribute that contains this location information.

TASK: Find the complete XML attribute tag that contains the location value "{location_value}"

RULES:
1. Return the COMPLETE attribute tag including all attributes and content
2. Look for geographic location attributes like: geo_loc_name, geographic location, country, latitude, longitude, etc.
3. The location value might be contained within the attribute content
4. If multiple attributes contain the value, return the most specific geographic one
5. If no matching attribute is found, return exactly: "NONE"

EXAMPLE OUTPUT FORMAT:
<Attribute attribute_name="geographic location (country and/or sea)" harmonized_name="geo_loc_name" display_name="geographic location">Iceland</Attribute>

XML BLOCK:
{block}

LOCATION VALUE TO FIND: {location_value}

MATCHING ATTRIBUTE:"""

        response = self.ask_ollama_with_tools(prompt, self.attr_model, [], accession, max_retries=2)
        if response and "response" in response:
            result = response["response"].strip()
            
            # Nettoyer la réponse
            result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
            result = result.strip()
            
            if result and result.upper() != "NONE" and "<Attribute" in result:
                pattern = r'<Attribute\b[^>]*>.*?</Attribute>'
                return re.findall(pattern, result, flags=re.DOTALL)
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
    def resolve_place_to_coordinates(self, place_name: str, accession: str) -> Optional[str]:
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
            resp = self.ask_ollama_with_tools(prompt, self.coord_model, tools=[llm_geocode_tool], accession=accession)
            
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
        """Traitement complet d'un bloc BioSample avec extraction d'attributs sources"""
        result = LocationResult(accession=accession)
        
        # Étape 1: Extraction par regex
        regex_coords, regex_source = self.extract_coordinates_regex(block)
        if regex_coords:
            result.direct_location = regex_coords
            result.direct_method = "regex"
            result.final_location = regex_coords
            result.processing_method = "regex"
            result.source_attribute = regex_source
            logger.debug(f"{accession}: Found coordinates via regex: {regex_coords}")
            return result

        # Étape 2: Tentative HapMap
        hapmap_coords, hapmap_source = self.extract_hapmap_location(block)
        if hapmap_coords:
            result.direct_location = hapmap_coords
            result.direct_method = "hapmap"
            result.final_location = hapmap_coords
            result.processing_method = "hapmap"
            result.source_attribute = hapmap_source
            logger.debug(f"{accession}: Found coordinates via HapMap: {hapmap_coords}")
            return result

        # Étape 3: Extraction LLM
        result.llm_location = self.llm_extract_location(block, accession)
        if result.llm_location:
            # Tentative de résolution en coordonnées
            if self.is_valid_coordinate(result.llm_location):
                result.final_location = result.llm_location
                result.processing_method = "llm_direct"
                logger.debug(f"{accession}: LLM returned direct coordinates: {result.llm_location}")
            else:
                # Résolution de lieu en coordonnées
                result.resolved_coordinates = self.resolve_place_to_coordinates(result.llm_location, accession)
                if result.resolved_coordinates:
                    result.final_location = result.resolved_coordinates
                    result.processing_method = "llm_resolved"
                    logger.debug(f"{accession}: Resolved '{result.llm_location}' to {result.resolved_coordinates}")
                else:
                    # Garder le nom de lieu comme résultat final
                    result.final_location = result.llm_location
                    result.processing_method = "llm_place"
                    logger.debug(f"{accession}: Kept place name: {result.llm_location}")
            
            # Extraction de l'attribut source pour la localisation LLM
            if result.final_location:
                result.source_attribute = self.extract_source_attribute(block, result.llm_location, accession)
            
            
        return result

    def __del__(self):
        """Ferme proprement le fichier de log à la destruction"""
        if hasattr(self, 'llm_logger') and self.llm_logger:
            try:
                self.llm_logger["file"].close()
            except Exception as e:
                logger.error(f"Error closing LLM log file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Extract location data from SRA XML with source attribution.")
    parser.add_argument("file", help="Input XML file path")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama model name for general processing")
    parser.add_argument("--coord_model", default="mistral", help="Ollama model for coordinate resolution")
    parser.add_argument("--attr_model", default="mistral", help="Ollama model for attribute extraction")
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
    extractor = LocationExtractor(args.model, args.coord_model, args.attr_model, args.hapmap)
    
    # Extraction des blocs BioSample
    biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    logger.info(f"Found {len(biosample_blocks)} BioSample blocks in {args.file}")

    if not biosample_blocks:
        logger.warning("No BioSample blocks found in the input file")
        sys.exit(1)

    # Traitement et écriture des résultats
    with open(args.output, "w", newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "Accession", "Final_Location", "Processing_Method", 
            "Direct_Location", "Direct_Method", "LLM_Location", 
            "Resolved_Coordinates", "Source_Attribute"
        ])

        start_time = time.time()
        processed = 0
        successful_extractions = 0

        for i, block in enumerate(tqdm(biosample_blocks, desc="Processing BioSamples"), 1):
            accession_match = re.search(r'accession="([^"]+)"', block)
            accession = accession_match.group(1) if accession_match else f"UNKNOWN_{i}"

            result = extractor.process_biosample_block(block, accession)

            writer.writerow([
                result.accession,
                result.final_location or "",
                result.processing_method or "",
                result.direct_location or "",
                result.direct_method or "",
                result.llm_location or "",
                result.resolved_coordinates or "",
                json.dumps(result.source_attribute) if result.source_attribute else ""
            ])

            if result.final_location:
                successful_extractions += 1

            processed += 1

            # Optionally update tqdm postfix
            if i % args.batch_size == 0 or i == len(biosample_blocks):
                elapsed = time.time() - start_time
                avg_time = elapsed / processed if processed > 0 else 0
                success_rate = (successful_extractions / processed * 100) if processed > 0 else 0

                tqdm.write(f"Progress: {i}/{len(biosample_blocks)} blocks processed "
                        f"({success_rate:.1f}% success rate, {avg_time:.2f}s/block)")

        # Final stats
        total_time = time.time() - start_time
        logger.info(f"Processing complete! {successful_extractions}/{processed} successful extractions "
                f"({successful_extractions/processed*100:.1f}% success rate)")
        logger.info(f"Total processing time: {total_time:.2f}s, Average: {total_time/processed:.2f}s/block")
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()