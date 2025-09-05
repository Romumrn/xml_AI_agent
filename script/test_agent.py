import ollama
from geopy.geocoders import Nominatim
from pprint import pprint
import sys
import re

# Ton outil Python réel
def get_coordinate(place):
    app = Nominatim(user_agent="tutorial")
    location = app.geocode(place, timeout=10)
    if location:
        return {"latitude": location.latitude, "longitude": location.longitude, "LLM_place_found": place}
    else:
        return "NA"
    
def ask_agent( block_xml, model ):
    response = ollama.chat(
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
    print("REPONSE BRUTE:", response)

    if response.message.tool_calls:
        results = []
        for call in response.message.tool_calls:
            # if call.function.name == "get_coordinate":
            #     place = call.function.arguments["place_name"]
            #     result = get_coordinate(place)
            #     print( result)
            #     # Étape 3 : On redonne le résultat à CE MÊME modèle
            #     # Pas forcement utile 
            #     followup = ollama.chat(
            #         model="mistral",
            #         messages=[
            #             {"role": "assistant", "content": response.message.content},
            #             {"role": "tool", "content": str(result)},
            #             {"role": "user", "content": "Output ONLY a JSON object like {\"latitude\": X.XXXX, \"longitude\": Y.YYYY, \"place found\": PLACE} or NA."}
            #         ]
            #     )
            # for call in response.message.tool_calls:
            if call.function.name == "get_coordinate":
                place = call.function.arguments["place_name"]
                result = get_coordinate(place)
                print(f"Geocoding result for '{place}':", result)
                results.append(result)
            
            elif call.function.name == "check_coordinate":
                coord = call.function.arguments["coordinate"]
                result = check_coordinate(coord)
                print(f"Coordinate validation for '{coord}':", result)
                results.append(result)

        return results
    else:
        return "NA"
    
def check_coordinate(coord: str) -> bool:
        
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
            # DMS
            r'^-?\d+°\d+\'\d*\.?\d*"?\s*[NS],?\s*-?\d+°\d+\'\d*\.?\d*"?\s*[EW]$',
            # Décimal avec direction
            r'^-?\d+\.\d+\s*[NS],?\s*-?\d+\.\d+\s*[EW]$',
        ]

        return any(re.match(pattern, coord, re.IGNORECASE) for pattern in coord_patterns)

 
def process_biosamples( biosamples_filename, model ):
    try:
        with open(biosamples_filename, "r", encoding="utf-8") as f:
            xml_data = f.read()
    except FileNotFoundError:
        print(f"Input file {biosamples_filename} not found")
        sys.exit(1)
        
    # Extraction des blocs BioSample
    biosample_blocks = re.findall(
        r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)
    print( f"Found {len(biosample_blocks)} BioSample blocks in {biosamples_filename}")

    if not biosample_blocks:
        print("No BioSample blocks found in the input file")
        sys.exit(1)
        
    for i, block in enumerate(biosample_blocks, start=1):
            accession_match = re.search(r'accession="([^"]+)"', block)
            accession = accession_match.group(1) if accession_match else f"UNKNOWN_{i}"
            result = ask_agent(block, model)
            print(accession, model, result, "\n")
    
for model in ['llama3.1:8b', 'gpt-oss:20b']:
    process_biosamples( sys.argv[1], model)
        


    
