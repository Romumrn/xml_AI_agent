
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



## Example XML 

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
...
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



**Example:**
```csv
SAMN21498129,gpt-oss:20b,23.0195,113.4100,"Jinan University",0.33
```


##  Evaluation Pipeline

A dedicated script benchmarks multiple models and compares their results against ground truth.

### Run all models + evaluate

```bash
python batch_llm_eval.py
```

## Benchmark Results

### Quality Metrics Comparison

| Model              | Accuracy | Precision | Recall | F1-Score | Specificity |
|--------------------|----------|-----------|--------|----------|-------------|
| **gpt-oss:20b**    | 53.3%    | **81.8%** | 42.9%  | **56.3%**| **77.8%**   |
| **mistral-nemo:12b**| 46.7%   | 52.9%     | **52.9%**| 52.9%  | 38.5%       |
| **qwen3:8b**       | 43.3%    | **100.0%**| 22.7%  | 37.0%    | **100.0%**  |
| **cogito:8b**      | 40.0%    | **100.0%**| 18.2%  | 30.8%    | **100.0%**  |
| **granite4:3b**    | 33.3%    | 35.7%     | 31.3%  | 33.3%    | 35.7%       |
| **llama3.1:8b**    | 26.7%    | 25.0%     | 28.6%  | 26.7%    | 25.0%       |

#### Confusion Matrix Summary

| Model              | TP | TN | FP | FN |
|--------------------|----|----|----|----|
| **gpt-oss:20b**    | 9  | 7  | 2  | 12 |
| **mistral-nemo:12b**| 9  | 5  | 8  | 8  |
| **qwen3:8b**       | 5  | 8  | 0  | 17 |
| **cogito:8b**      | 4  | 8  | 0  | 18 |
| **granite4:3b**    | 5  | 5  | 9  | 11 |
| **llama3.1:8b**    | 4  | 4  | 12 | 10 |

**Legend:** TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives


### Performance Metrics

| Model              | Mean Time (s) | Median Time (s) | P95 Time (s) | Total Time (s) |
|--------------------|---------------|-----------------|--------------|----------------|
| **granite4:3b**    | **3.18**      | **2.90**        | **4.40**     | **95.4**       |
| **mistral-nemo:12b**| 7.50         | 6.87            | 9.99         | 224.8          |
| **llama3.1:8b**    | 9.03          | 4.80            | 13.62        | 270.9          |
| **cogito:8b**      | 9.24          | 8.36            | 14.37        | 277.1          |
| **gpt-oss:20b**    | 11.98         | 9.42            | 24.05        | 359.3          |
| **qwen3:8b**       | 36.56         | 32.46           | 60.03        | 1096.9         |


### Key Findings

**🏆 Best Overall Model: gpt-oss:20b**
- Highest F1-score (56.3%) and accuracy (53.3%)
- Excellent precision (81.8%) with acceptable recall
- Moderate processing speed (~12s per sample)

**⚡ Fastest Model: granite4:3b**
- 3.18s average per sample (3.8x faster than gpt-oss)
- Trade-off: Lower accuracy (33.3%)

**🎯 Most Conservative: qwen3:8b & cogito:8b**
- Perfect precision (100%) but very low recall (<23%)
- Avoid false positives but miss many valid locations

**⚠️ Timeout Issues: qwen3:8b**
- 9 timeouts out of 30 samples
- Slowest overall (36.6s average)

---

## 🔬 Alternative Approaches

### GLiNER (Zero-shot NER)

```bash
python script/run_with_gliner.py --input data_test/Dataset_test_biosample.xml
```

**Results:**
- **Accuracy:** 33.3%
- **Precision:** 66.7%
- **Recall:** 18.2%
- **F1-Score:** 28.6%
- **Speed:** 0.75s per sample (16x faster than gpt-oss)
- **Success rate:** 6/30 samples geolocalized

### spaCy (Transformer-based NER)

```bash
python script/run_with_spacy.py --input data_test/Dataset_test_biosample.xml
```

**Results:**
- **Accuracy:** 33.3%
- **Precision:** 50.0%
- **Recall:** 20.0%
- **F1-Score:** 28.6%
- **Speed:** 0.58s per sample (20x faster than gpt-oss)
- **Success rate:** 8/30 samples geolocalized



## Comparison Summary

| Method            | F1-Score | Speed (s) | Best For                    |
|-------------------|----------|-----------|------------------------------|
| **gpt-oss:20b**   | 56.3%    | 11.98     | Highest quality extraction   |
| **mistral-nemo**  | 52.9%    | 7.50      | Balanced quality/speed       |
| **granite4:3b**   | 33.3%    | 3.18      | High-volume, speed priority  |
| **spaCy**         | 28.6%    | 0.58      | Ultra-fast baseline          |
| **GLiNER**        | 28.6%    | 0.75      | Zero-shot, no training       |

---

## Conclusion

For the **Virome@tlas** pipeline, **gpt-oss:20b** provides the best balance of accuracy and reliability for geographic extraction from messy SRA metadata. For large-scale processing where speed is critical, **mistral-nemo:12b** offers a good compromise, while traditional NER approaches (spaCy, GLiNER) remain too limited for this complex task.
