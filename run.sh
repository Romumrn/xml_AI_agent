for model in "gemma3:4b" "mistral:latest" "phi4:latest" "gemma3:12b"; do
  outfile="test353_${model//:/_}.csv"
  echo "[INFO] Running model: $model -> $outfile"
  time python script/llm_agent_api.py /home/ubuntu/random_1000_biosamples.xml --model "$model" --output "$outfile"
done

python - <<EOF
import pandas as pd

# Helper to load, filter, and rename
def load_and_prepare(path, model_name):
    df = pd.read_csv(path, dtype=str)
    # Keep accession + LLM prompt outputs
    df = df[['Accession', 'LLM_loc_1', 'LLM_loc_2']]
    df = df[df['LLM_loc_1'].notna() | df['LLM_loc_2'].notna()]  # Keep rows with at least one prompt result
    df = df.rename(columns={
        'LLM_loc_1': f'LLM_loc_{model_name}_P1',
        'LLM_loc_2': f'LLM_loc_{model_name}_P2'
    })
    return df

# Load and prepare each file (correct filenames)
gemma3 = load_and_prepare('test353_gemma3_4b.csv', 'Gemma3_4b')
gemma3_12b = load_and_prepare('test353_gemma3_12b.csv', 'Gemma3_12b')
mistral = load_and_prepare('test353_mistral_latest.csv', 'Mistral')
phi4 = load_and_prepare('test353_phi4_latest.csv', 'Phi4')

# Merge on Accession
df = gemma3.merge(gemma3_12b, on='Accession', how='outer') \
           .merge(mistral, on='Accession', how='outer') \
           .merge(phi4, on='Accession', how='outer')

# Save merged output
df.to_csv('test353_merged.csv', index=False)
print("[INFO] Merged CSV written to test353_merged.csv")

EOF
