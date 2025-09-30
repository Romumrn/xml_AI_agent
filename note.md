# Notes sur le développement de l’agent

## Différents tests

* **Ollama vs HuggingFace**
* Parler des **SLM** et des échecs avec exemples
* Tests avec **Transformers** et HuggingFace



## Tests classiques

* Expliquer pourquoi on faisait plusieurs appels au modèle pour obtenir une meilleure réponse.
* Transition vers un mode **agent** pour aller chercher directement les coordonnées.
* Pour l’exécution : il faut mettre une boucle qui appelle la fonction *tool*, puis renvoie la réponse.


## Vérifier l’exécution par un modèle

⚠️ Problème : certains modèles sont mauvais pour l’exécution d’agent et ne renvoient rien dans `tool_calls`.

Exemple de structure attendue :

```python
tool_calls = [
  ToolCall(
    function=Function(
      name='get_coordinate',
      arguments={'place_name': 'San Francisco'}
    )
  )
]
```

Cas concrets :

* **Mistral** → renvoie toujours `None`
* **Ollama 3.1** → réussit bien

> "Mistral 0.3 supports function calling with Ollama’s raw mode."
> 🔗 [https://ollama.com/library/mistral](https://ollama.com/library/mistral)


## Problème d’invention d’attributs

Une partie du script extrait l’attribut XML contenant l’information géographique.
Problème : le modèle (ex. Mistral) **invente des attributs** qui n’existent pas.

### Exemple d’entrée (BioSample)

```xml
<BioSample ...>
   ...
   <Attribute attribute_name="sampling site">femur</Attribute>
   ...
</BioSample>
```

### Résultat obtenu (erroné)

```
SAMEA6841538,"Cambridge, UK
femur",llm_place,,,"Cambridge, UK
femur",,"[""<Attribute attribute_name=\""sampling site\"" harmonized_name=\""sampling_site\"" display_name=\""sampling site\"">Cambridge, UK femur</Attribute>""]"
```

👉 Ici, le modèle a combiné la localisation "Cambridge, UK" avec "Femur" qui n’était pas dans les données d’origine.


## Difficulté : homogénéiser le prompt pour différents modèles

Comparaison des valeurs de réponses pour **le même échantillon avec le même prompt**.

### Prompt utilisé

```python
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
```

---

### Comparaison des résultats

**Avec Llama3.1:8b**

```
SAMN18357804,llama3.1:8b,2025-09-01 12:13:23,
"Most specific geographic identifier: State Key Laboratory of Microbial Metabolism, Room 3517, Biology Building, 837 Dongchuang Road, Shanghai, China
Latitude/longitude: Not available
City: Shanghai
Country: China"
```

**Avec Gemma3:12b**

```
SAMN18357804,gemma3:12b,2025-09-01 12:21:03,"Shanghai, China"
```


**Avec un SLM**


Definition:
Compact models (<10B parameters) optimized for low latency, local use, fast fine-tuning
How to build a SLM:
Train from scratch (rare, costly)
Distillation from larger models
Pruning & Quantization (e.g., 4/8-bit)
Adapters (e.g., LoRA)
Advantages:
Runs on small GPU
Offline/private use
Energy efficient
TinyLlama  (1,1B)

Create a Virtual Environment
Set up a Python venv & load required packages: transformers, accelerate,, etc.
Authenticate Hugging Face CLI 
Run: huggingface-cli login & enter your token to access gated models
Load Model with Transformers directly from Hugging Face
Prepare XML Data & Generate Responses
Load and parse large XML input & Split into smaller <BioSample> blocks for manageable prompts
Feed each XML chunk into the model via prompt
Collect and process model outputs in desired format (e.g. CSV)

reponse 
```
To extract only geographic location information from the XML block, use the function isNeeded() in Python. Here\'s an example:\n\n```python\nimport requests\nfrom lxml import html\n\nurl = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5084390"\nresponse = requests.get(url)\nhtml_doc = html.fromstring(response.content)\n\nlocations = []\nfor node in html_doc.xpath("//div[@class=\'gene-info\']/text()", namespaces={\'\': \'http://www.ncbi.nlm.nih.gov/genome/gene/feature/\'}):\n    if node.text.startswith(\'Latitude\'):\n        locations.append(node.text)\n\nprint(locations)\n```\n\nThis code uses the `lxml` library to extract the text nodes with the class `gene-info` and the attribute `text()` that contains the latitude information. The function `isNeeded()` checks if the element has a specific attribute or class, which is used to filter out irrelevant elements.
```



## Problème de fausse localisation

Exemple d’erreur :

```
SAMN14327956,"37.4000967,-121.9397177",llm_resolved,,,
"Lung, USA","37.4000967,-121.9397177",
"[""<Attribute attribute_name=\""body site\"" harmonized_name=\""tissue\"" display_name=\""tissue\"">Lung</Attribute>""]"
```


## Réponse non homogène

Le même script donne deux sorties différentes pour le même appel.

### Exemple :


```
(venv) ubuntu@machine7c531140-edae-4a80-a530-45ade815e99e:~/xml_AI_agent$ python script/test_agenbt.py 
model='llama3.1:8b' created_at='2025-09-02T07:50:41.257082136Z' done=True done_reason='stop' total_duration=548702760 load_duration=175451928 prompt_eval_count=828 prompt_eval_duration=23525897 eval_count=21 eval_duration=347787817 message=Message(role='assistant', content='', thinking=None, images=None, tool_calls=[ToolCall(function=Function(name='get_coordinate', arguments={'place_name': 'Uppsala University'}))])
 This is the location information for Uppsala University, Sankt Olofsgatan, Främre Luthagen, Fjärdingen, Uppsala, Uppsala kommun, Uppsala län, 753 11, Sverige.

Latitude: 59.8576363
Longitude: 17.6294616
Address: Uppsala universitet, Sankt Olofsgatan, Främre Luthagen, Fjärdingen, Uppsala, Uppsala kommun, Uppsala län, 753 11, Sverige

```
```
(venv) ubuntu@machine7c531140-edae-4a80-a530-45ade815e99e:~/xml_AI_agent$ python script/test_agenbt.py 
model='llama3.1:8b' created_at='2025-09-02T07:50:45.510914597Z' done=True done_reason='stop' total_duration=542034594 load_duration=178063992 prompt_eval_count=828 prompt_eval_duration=16192842 eval_count=21 eval_duration=345416007 message=Message(role='assistant', content='', thinking=None, images=None, tool_calls=[ToolCall(function=Function(name='get_coordinate', arguments={'place_name': 'Uppsala University'}))])
 The provided data represents the location of Uppsala University, which is situated at the following coordinates:

- Latitude: 59.8576363
- Longitude: 17.6294616

The complete address is:

Uppsala universitet, Sankt Olofsgatan, Främre Luthagen, Fjärdingen, Uppsala, Uppsala kommun, Uppsala län, 753 11, Sverige

This address is located in Uppsala, Sweden.
```

Dans un cas, l’outil est bien appelé ; dans l’autre, non. Ici le probleme a été partiellement reglé en fixant la teùmperature a 0, mais il subiste des differences. 


Realisation d'un jeu de donné test pour verifier les données, ici je vais recuperer les données du fichier biosamples xml pour ezliser un echantiollonage des erreur courante pour pouvoire realidser des scrore lors de chaque changement de prompt, model ou methode. 
Il va faloir verifier chaque Biosample avec zero localisation renvois bien NA ou Null, que chaque localisation renvois bien les bonne coordoonées ( scrore de distance ?), que les faux positif soit bien eliminer (Valeur de localisation pour des partie du corps par exemple) 



# Test Extraction de localisations aget avec Ollama + Tools

## Objectif

Extraire l’information géographique depuis des blocs XML en utilisant un LLM (Ollama) et des outils (`get_coordinate`, `check_coordinate`).

```text
SYSTEM:
You are an AI assistant specialized in extracting geographic location from XML.
Rules:
- Do NOT return code, explanations, or JSON inside "content".
- ALWAYS call the appropriate tool (get_coordinate or check_coordinate).
- If no geographic information is found, return "NA".

USER:
Extract ONLY the most specific geographic identifier from this XML block.
- If you find coordinates, call check_coordinate("lat,lon").
- If you find place names, call get_coordinate("City, Country").
- If nothing, return "NA".
```

## Observations

### llama3.1:8b

####  Comportement :

Mélange de code Python/JSON brut dans message.content (inexploitable).

Génère parfois des tool_calls, mais peu fiables.

####  Résultats :

Beaucoup de NA → incapacité à isoler correctement des lieux.

Quand ça marche, arrive à renvoyer des lieux plausibles (ex: Peru).

####  Limite :

N’arrive pas à séparer entités biologiques des lieux → faux positifs fréquents.

### gpt-oss:20b

#### Comportement :

Stable dans l’usage des tool_calls.

Fait des get_coordinate propres avec des lieux clairs (Davis, CA, Shenyang, China).

####  Résultats :

Bonne couverture : Davis, Mauritius, Peru, Shenyang, etc.

Parfois retourne directement des coordonnées déjà en entrée ("44.3599167,5.1302223").

Quelques NA quand le texte est bruité.

####  Limite :

Légèrement plus lent (temps moyen ≈ 5–10s).

Tendance à inventer des localisations approximatives (Shimla, India au lieu de Peru).

### qwen3:8b

####  Comportement :

Produit souvent des résultats, parfois plusieurs coordonnées pour un seul échantillon.

Plus bavard et incertain.

####  Résultats :

Correct pour UC Davis, Cambridge, Shenyang, Peru.

Beaucoup de temps de calcul (30–50s sur certains cas).

Résultats incohérents ou multiples (deux lat/lon pour SAMEA6018323).

####  Limite :

Trop lent pour du traitement massif.

Moins stable que gpt-oss:20b.

### deepseek-r1:latest

####  Comportement :

⚠️ Ne supporte pas les tools → tout est NA.

####  Résultats :

Toujours vide, aucun appel fonctionnel.

####  Limite :

Inutilisable dans ce pipeline.

### mistral-nemo:latest

####  Comportement :

Très efficace : produit de bons tool_calls et lieux pertinents.

####  Résultats :

Repère correctement : Guangzhou, Michigan, Davis, Cambridge, Mauritius, Peru, New York, Qingdao.

Rapidité excellente (1–3s en moyenne).

Rarement des NA.

####  Limite :

Peut généraliser trop (ex: New York, USA pour plusieurs échantillons distincts).

### granite3.1-moe:3b

####  Comportement :

Quasi toujours NA.

####  Résultats :

Ne déduit rien d’exploitable.

####  Limite :

Trop faible pour l’extraction géographique.

🔹 phi4-mini:3.8b

####  Comportement :

Comme granite3.1-moe, retourne principalement NA.

####  Résultats :

Aucun lieu significatif détecté.

####  Limite :

Inutilisable pour ce cas.

### llama3-groq-tool-use:8b

####  Comportement :

Bien qu’indiqué comme tool-use, il n’arrive pas à générer de vrais tool_calls.

####  Résultats :

Toujours NA dans cette tâche.

####  Limite :

Mauvaise intégration avec Ollama tools.