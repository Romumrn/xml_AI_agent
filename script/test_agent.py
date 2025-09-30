import ollama
from geopy.geocoders import Nominatim
from pprint import pprint
import sys
import re
import csv
import time
import concurrent.futures
import statistics
import csv
import sys
import subprocess
import json

def check_and_pull_model(model):
    """
    Vérifie si un modèle Ollama est installé, sinon le télécharge
    
    Args:
        model (str): Nom du modèle Ollama
        
    Returns:
        bool: True si le modèle est prêt à être utilisé
    """
    try:
        # Lister les modèles installés
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extraire le nom du modèle (sans le tag si non spécifié)
        model_base = model.split(':')[0]
        
        # Vérifier si le modèle est dans la liste
        if model_base in result.stdout or model in result.stdout:
            print(f"✓ Modèle '{model}' déjà installé")
            return True
        else:
            print(f"⚠ Modèle '{model}' non trouvé, téléchargement en cours...")
            # Télécharger le modèle
            pull_result = subprocess.run(
                ["ollama", "pull", model],
                check=True
            )
            print(f"✓ Modèle '{model}' téléchargé avec succès")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors de la vérification/téléchargement du modèle '{model}': {e}")
        return False
    except FileNotFoundError:
        print("✗ Ollama n'est pas installé ou n'est pas dans le PATH")
        return False
    
def get_coordinate(place):
    app = Nominatim(user_agent="tutorial")
    location = app.geocode(place, timeout=10)
    if location:
        return {"latitude": location.latitude, "longitude": location.longitude, "LLM_place_found": place}
    else:
        return "NA"
    
def check_coordinate(coord: str):
    # Test de validation des valeurs numériques pour le format décimal
    if re.match(r'^-?\d+\.?\d*,\s*-?\d+\.?\d*$', coord):
        try:
            lat_str, lon_str = coord.split(',')
            lat, lon = float(lat_str.strip()), float(lon_str.strip())
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                # Return the coordinate as a dictionary instead of boolean
                return {"latitude": lat, "longitude": lon, "LLM_place_found": coord}
        except (ValueError, IndexError):
            pass
    
    # If coordinate is invalid, try to parse other formats
    # Add your other coordinate parsing logic here if needed
    return "NA"
import multiprocessing as mp

def _ollama_call_worker(model, block_xml):
    """
    Fonction exécutée dans un sous-processus pour éviter les blocages
    """
    return ollama.chat(
        model=model,
        options={'temperature': 0},
        messages=[{
            'role': 'user',
            'content': f'''
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

            XML BLOCK:
            {block_xml}
            '''
        }],
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': 'get_coordinate',
                    'description': 'Fetch decimal latitude,longitude for a place name',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'place_name': {
                                'type': 'string',
                                'description': 'A place to geocode',
                            },
                        },
                        'required': ['place_name'],
                    },
                },
            },
            {
                'type': 'function',
                'function': {
                    'name': 'check_coordinate',
                    'description': 'Check coordinate format: latitude,longitude and return validation info',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'coordinate': {
                                'type': 'string',
                                'description': 'latitude,longitude string to validate',
                            },
                        },
                        'required': ['coordinate'],
                    },
                },
            },
        ],
    )
    
def ask_agent(block_xml, model, timeout=50):
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        result = pool.apply_async(_ollama_call_worker, (model, block_xml))
        try:
            response = result.get(timeout=timeout)
        except mp.TimeoutError:
            pool.terminate()
            print(f"[TIMEOUT] Ollama a dépassé {timeout}s pour le modèle {model}")
            return "NA"
        except Exception as e:
            pool.terminate()
            print(f"[ERREUR] lors de l'appel Ollama : {e}")
            return "NA"

    # Même logique de traitement que dans ton code actuel
    if response.message.tool_calls:
        results = []
        for call in response.message.tool_calls:
            if call.function.name == "get_coordinate":
                place = call.function.arguments["place_name"]
                results.append(get_coordinate(place))
            elif call.function.name == "check_coordinate":
                coord = call.function.arguments["coordinate"]
                results.append(check_coordinate(coord))
        return results
    else:
        print(response)
        return "NA"
def process_biosamples(biosamples_filename, model, writer_output_file):
    try:
        with open(biosamples_filename, "r", encoding="utf-8") as f:
            xml_data = f.read()
    except FileNotFoundError:
        print(f"Input file {biosamples_filename} not found")
        sys.exit(1)
        
    # Extraction des blocs BioSample
    biosample_blocks = re.findall(
        r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    print(f"Found {len(biosample_blocks)} BioSample blocks in {biosamples_filename}")

    if not biosample_blocks:
        print("No BioSample blocks found in the input file")
        sys.exit(1)
        
    c = 0
    list_execution_time = []
    for i, block in enumerate(biosample_blocks, start=1):
        start_time = time.time()
        accession_match = re.search(r'accession="([^"]+)"', block)
        accession = accession_match.group(1) if accession_match else f"UNKNOWN_{i}"
        
        result = ask_agent(block, model)
        
        # IMPORTANT: Calculer execution_time ICI, juste après ask_agent
        end_time = time.time()
        execution_time = end_time - start_time
        list_execution_time.append(execution_time)
        
        print(f"#{c}")
        if result == "NA":
            writer_output_file.writerow({
                "accession": accession,
                "model": model,
                "latitude": "",
                "longitude": "",
                "place": result,
                "execution_time": f"{execution_time:.2f}"
            })
            
            print(c, {
                "accession": accession,
                "model": model,
                "latitude": "",
                "longitude": "",
                "place": result,
                "execution_time": f"{execution_time:.2f}"
            })
        
        else:
            for res in result:
                if res != "NA":  # Only write valid results
                    writer_output_file.writerow({
                        "accession": accession,
                        "model": model,
                        "latitude": res["latitude"],
                        "longitude": res["longitude"],
                        "place": res["LLM_place_found"],
                        "execution_time": f"{execution_time:.2f}"
                    })
                    print(c, {
                        "accession": accession,
                        "model": model,
                        "latitude": res["latitude"],
                        "longitude": res["longitude"],
                        "place": res["LLM_place_found"],
                        "execution_time": f"{execution_time:.2f}"
                    })
        c += 1
        print(f"{accession} - Résultat: {result} - Temps d'exécution: {execution_time:.2f} secondes")
    
    if list_execution_time:
        print(f"DONE - Moyenne: {statistics.mean(list_execution_time):.2f}s")
    else:
        print("DONE - Aucun temps d'exécution enregistré")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # utile surtout sous Windows, inoffensif ailleurs

    output_file = "biosample_results.csv"        
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["accession", "model", "latitude", "longitude", "place", "execution_time"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if f.tell() == 0:
            writer.writeheader()

        for model in [
            'llama3.1:8b',
            'gpt-oss:20b',
            'qwen3:8b',
            "deepseek-r1:latest",
            "mistral-nemo:latest",
            "granite3.1-moe:3b",
            "phi4-mini:3.8b",
            "llama3-groq-tool-use:8b",
        ]:
            if check_and_pull_model(model):
                try:
                    process_biosamples(sys.argv[1], model, writer)
                except Exception as e:
                    print(f"✗ Erreur lors du traitement avec le modèle '{model}': {e}")
                    continue
            else:
                print(f"⊗ Modèle '{model}' ignoré (échec de l'installation)")
                continue
