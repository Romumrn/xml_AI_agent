
# **xml\_AI\_agent**

### **Context**

**Virome\@tlas** aims to build a global cloud-based platform for viral surveillance, integrating large-scale sequencing and geographic data to monitor virus diversity, hosts, and environments.
As part of this project, we focus on parsing massive SRA files to extract meaningful information.


### ❗ **Problem**

SRA (`.xml`) files often contain poorly annotated fields, especially the location. 


### 🔎 **Example data**

```xml
<Attribute attribute_name="Longitude">60 16' 10'' N</Attribute>
<Attribute attribute_name="Latitude">5 13' 20'' E</Attribute>
```

```xml
<Attribute attribute_name="lat_lon" harmonized_name="lat_lon" display_name="latitude and longitude">32.167 N 64.50 W</Attribute>
```

```xml
<Attribute attribute_name="geo_loc_name" harmonized_name="geo_loc_name" display_name="geographic location">USA: Oregon</Attribute>
```

```xml
<Attribute attribute_name="geo_loc_name" harmonized_name="geo_loc_name" display_name="geographic location">Kenya: Shimoni</Attribute>
<Attribute attribute_name="lat_lon" harmonized_name="lat_lon" display_name="latitude and longitude">NA</Attribute>
```

But usally the location is not provided of in the wrong field. 


### 💡 **Idea**

Use a **Large Language Model (LLM)** to automatically extract this information from XML files.


## 🤖 **Why use an LLM?**

A **Large Language Model (LLM)** is trained on vast amounts of text using self-supervised learning. It is designed for natural language processing tasks such as text generation, summarization, and classification.

✅ **Advantages**

* Automates complex tasks (summarization, classification, content generation)
* Improves productivity and writing quality
* Enables interaction via natural language

⚠ **Disadvantages**

* High environmental impact (hardware, training)
* Potential for biased outputs
* Data privacy and security risks (especially with private services like ChatGPT)
* Requires expertise and resources (for local deployment)


## Process (to be changed) 

**Using an LLM (via Ollama)**

Attempt to extract location data from `<BioSample>` using:

* **Transformers**
* **Python**
* **Hugging Face**


1️⃣ **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate
```

2️⃣ **Install required packages**

```bash
pip install transformers accelerate huggingface_hub
```

3️⃣ **Authenticate Hugging Face CLI**

```bash
huggingface-cli login
```

(Enter your token to access gated models)

4️⃣ **Load the model**

* Use `transformers` to load the model directly from Hugging Face.

5️⃣ **Prepare XML data**

* Load and parse large XML files
* Split into smaller `<BioSample>` blocks for manageable prompts
* Feed the XML chunks to the model
* Collect and process the outputs (e.g., CSV)