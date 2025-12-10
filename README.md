
# **xml_AI_agent — LLM-Based Geographic Extraction from SRA XML**

## **Context**

**Virome@tlas** is a global cloud platform for viral surveillance, integrating large-scale sequencing datasets with geographic metadata to better understand virus diversity, host ecology, and environmental context. A major component of the pipeline is extracting precise geographic locations from SRA BioSample XML files.

While SRA metadata contains potentially valuable location information, it is often **incomplete, inconsistent, or deeply embedded in unstructured text**. This project provides a fully automated agent-powered pipeline to extract, validate, standardize, and geocode geographic information from raw XML data.

---

# **The Problem**

SRA XML metadata suffers from:

* **Inconsistent formats**

  * `"USA: Oregon"`
  * `"40° 12' 10'' N 72° W"`
  * `"32.167 N 64.50 W"`
* **Missing or incorrect fields**

  * `"NA"`
  * wrong field for latitude/longitude
* **Location embedded in unstructured text**

  * `"Collected near Lake George, Uganda during expedition..."`
* **Institution names used as proxies**

  * `"University of California, Davis"`
* **Complex biological metadata mixed with geographic info**
* **Population codes**

  * `"CEU"`, `"TSI"`, etc. (not handled yet)

Classical regex-based extraction cannot reliably handle this diversity.
**Large Language Models (LLMs) can.**

---

# **Solution: Agentic LLM-Based Geographic Extraction**

This project uses a **local LLM controlled by an agent** to interpret and extract geographic information from BioSample XML.
The agent collaborates with two executable tools:

### 🔧 **1. `check_coordinate(coord)`**

Validates latitude/longitude extracted from raw text.

### 🌍 **2. `get_coordinate(place)`**

Geocodes place names into decimal coordinates (Nominatim).

# **Agent Architecture**

```
BioSample XML Block
        ↓
   LLM Agent (Ollama)
        ↓
Parses → identifies → decides to call tools  
        ↓
 ┌───────────────────────────────┐
 │ Tool Calls (function calling) │
 │  - check_coordinate            │
 │  - get_coordinate              │
 └───────────────────────────────┘
        ↓
 Normalization & JSON repair
        ↓
 Final structured geographic output
```

### Key capabilities

* Extract the **most specific geographic entity** (city > region > country)
* Validate any coordinate structure
* Convert place names to canonical coordinates
* Output **strict JSON**, even when LLM output is messy
* Process **each BioSample independently in a spawned subprocess**
  → Stability + parallelism-friendly

---

# **Example XML **

### Coordinates in mixed formats

```xml
<Attribute attribute_name="Longitude">60 16' 10'' N</Attribute>
<Attribute attribute_name="Latitude">5 13' 20'' E</Attribute>
```

### Combined coordinate fields

```xml
<Attribute attribute_name="lat_lon">32.167 N 64.50 W</Attribute>
```

### Geographic names

```xml
<Attribute attribute_name="geo_loc_name">USA: Oregon</Attribute>
```

### Institutions / landmarks

```xml
<Attribute attribute_name="isolation_source">University of Michigan</Attribute>
```

### Missing or malformed fields

```xml
<Attribute attribute_name="lat_lon">NA</Attribute>
```

# 🚀 **Quick Start**

## 1. Install Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

## 2. Pull models

```bash
ollama pull llama3.1:8b
ollama pull gpt-oss:20b
ollama pull mistral-nemo:12b
```

## 3. Create environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Run extraction

```bash
python script/llm_agent_api.py \
  --input data_test/Dataset_test_biosample.xml \
  --model gpt-oss:20b \
  --output biosample_gptoss.csv
```


# **Output Format**

| Column           | Description                     |
| ---------------- | ------------------------------- |
| `accession`      | SRA BioSample accession         |
| `model`          | LLM used                        |
| `latitude`       | Decimal latitude (if available) |
| `longitude`      | Decimal longitude               |
| `place`          | Most specific name extracted    |
| `execution_time` | Runtime per block               |

Example:

```
SAMN21498129,gpt-oss:20b,23.0195,113.4100,"Jinan University",0.33
```

---

# 🧪 **Evaluation Pipeline**

A dedicated script benchmarks multiple models and compares their results against ground truth.

### Run all models + evaluate

```bash
python batch_llm_eval.py
```
... 

# Futur

* Improve Prompt
* Validate result 
* ...
