# **xml_AI_agent**

### **Context**

**Virome@tlas** aims to build a global cloud-based platform for viral surveillance, integrating large-scale sequencing and geographic data to monitor virus diversity, hosts, and environments. As part of this project, we focus on parsing massive SRA files to extract meaningful geographic information for epidemiological analysis.

### ❗ **Problem**

SRA (`.xml`) files often contain poorly annotated or inconsistent geographic location fields. Location information may be:
- **Missing or incomplete** (e.g., "NA", "not provided")
- **Incorrectly formatted** (e.g., coordinates in wrong fields)
- **Inconsistent formats** (e.g., decimal degrees vs. DMS notation)
- **Mixed with other data** (e.g., coordinates embedded in text)
- **Encoded as population codes** (e.g., HapMap population identifiers)

### 🔎 **Example data**

The tool handles various location formats found in SRA XML files:

**Coordinates with mixed latitude/longitude:**
```xml
<Attribute attribute_name="Longitude">60 16' 10'' N</Attribute>
<Attribute attribute_name="Latitude">5 13' 20'' E</Attribute>
```

**Combined coordinate fields:**
```xml
<Attribute attribute_name="lat_lon" harmonized_name="lat_lon" display_name="latitude and longitude">32.167 N 64.50 W</Attribute>
```

**Geographic location names:**
```xml
<Attribute attribute_name="geo_loc_name" harmonized_name="geo_loc_name" display_name="geographic location">USA: Oregon</Attribute>
```

**Incomplete or missing data:**
```xml
<Attribute attribute_name="geo_loc_name" harmonized_name="geo_loc_name" display_name="geographic location">Kenya: Shimoni</Attribute>
<Attribute attribute_name="lat_lon" harmonized_name="lat_lon" display_name="latitude and longitude">NA</Attribute>
```

**Population codes (HapMap):**
```xml
<Attribute attribute_name="population">CEU</Attribute>
```

### 💡 **Solution**

This project uses a **multi-layered approach** combining:
1. **Regex pattern matching** for structured coordinate formats
2. **HapMap population lookup** for known population codes (???)
3. **Large Language Model (LLM)** for complex text analysis
4. **Geocoding services** for place name resolution
5. **Coordinate validation** for data quality assurance

## 🤖 **Why use an LLM?**

Large Language Models excel at understanding context and extracting structured information from unstructured text, making them ideal for parsing inconsistent XML annotations.

✅ **Advantages**
* Handles complex, inconsistent data formats
* Understands context and relationships between fields
* Automates complex extraction tasks
* Adapts to various annotation styles
* Enables natural language interaction

⚠ **Considerations**
* Requires local deployment (Ollama) for data privacy
* Processing time depends on model size and complexity
* May require fine-tuning for specific domains
* Resource-intensive for large datasets

## 🏗️ **Architecture**
(LATER)

## 🚀 **Quick Start**

### Prerequisites

1. **Install Ollama** (for local LLM deployment)
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. **Pull required models**
```bash
ollama pull qwen2.5:7b
ollama pull llama3.1
ollama pull ...
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Romumrn/xml_AI_agent
cd xml_AI_agent
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Extract locations from SRA XML file
python script/llm_agent_api.py SRA_biosamples.xml --model gemma3:12b
```

## 📊 **Processing Pipeline**

The tool follows a hierarchical extraction strategy:

1. **Direct Extraction** (Regex + HapMap)
   - Pattern matching for coordinate formats
   - HapMap population code lookup
   - Fast, deterministic results

2. **LLM Analysis** (when direct methods fail)
   - Context-aware text analysis
   - Extraction of place names and coordinates
   - Handling of complex, unstructured data

3. **Geocoding Resolution**
   - Place name to coordinate conversion
   - Validation and standardization
   - Geographic data enrichment

4. **Quality Assurance**
   - Coordinate validation (-90/90, -180/180)
   - Format standardization
   - Source attribution tracking

## 📈 **Output Format**

Results are saved as CSV with the following columns:

| Column | Description |
|--------|-------------|
| `Accession` | SRA accession number |
| `Final_Location` | Best available location (coordinates or place name) |
| `Processing_Method` | Method used (regex, hapmap, llm_direct, llm_resolved, etc.) |
| `Direct_Location` | Result from direct extraction (regex/hapmap) |
| `Direct_Method` | Direct extraction method used |
| `LLM_Location` | LLM-extracted location |
| `Resolved_Coordinates` | Geocoded coordinates from place names |
| `Source_Attribute` | Source XML attribute(s) |


## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 **Known Issues & Limitations**

- Large XML files may require significant processing time
- LLM accuracy depends on model quality and prompt engineering
- Geocoding services have rate limits (1 request/second for Nominatim)
- Some complex coordinate formats may not be recognized
- HapMap coverage is limited to specific populations

## 📊 **Performance Metrics**

...
## 🔮 **Future Enhancements**

...