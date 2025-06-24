import re
import random
from xml.etree import ElementTree as ET
from xml.dom import minidom

def clean_whitespace(elem):
    """Remove empty whitespace nodes from XML tree"""
    for e in elem.iter():
        if e.text and not e.text.strip():
            e.text = None
        if e.tail and not e.tail.strip():
            e.tail = None
    return elem

def extract_random_biosamples(input_file, output_file, n=1000):
    # Read and preprocess the file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract BioSample entries
    biosamples = re.findall(r'(<BioSample[\s\S]*?</BioSample>)', content)
    if not biosamples:
        raise ValueError("No BioSample entries found")
    
    # Random selection
    selected = random.sample(biosamples, min(n, len(biosamples)))
    
    # Rebuild XML structure
    xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
    xml_content += '<BioSampleSet>\n'
    xml_content += '\n'.join(selected) + '\n'
    xml_content += '</BioSampleSet>'
    
    # Parse and clean formatting
    try:
        root = ET.fromstring(xml_content)
        clean_whitespace(root)
        rough_string = ET.tostring(root, encoding='utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Remove extra empty lines from pretty printing
        pretty_xml = '\n'.join(line for line in pretty_xml.split('\n') if line.strip())
    except ET.ParseError:
        pretty_xml = xml_content
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)

if __name__ == "__main__":
    extract_random_biosamples(
        input_file="biosample_subsample.xml",
        output_file="random_1000_biosamples.xml",
        n=1000
    )