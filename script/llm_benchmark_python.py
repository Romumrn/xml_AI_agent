from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import time
import csv
import os

# Device config
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}, dtype: {dtype}")

model_ids = [
    "google/gemma-1.1-2b-it",                          # 2B chat-tuned model
    "HuggingFaceH4/zephyr-7b-beta",
    
    "Qwen/Qwen1.5-1.8B-Chat",                     
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",                 
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",              # 1.1B parameters
    "mistralai/Mistral-7B-Instruct-v0.3",
]




# Load XML and find BioSample blocks
with open("test.xml", "r", encoding="utf-8") as f:
    xml_data = f.read()

biosample_blocks = re.findall(r"<BioSample.*?</BioSample>", xml_data, re.DOTALL)

# CSV output file
output_file = "model_benchmark_results.csv"
write_header = not os.path.exists(output_file)

with open(output_file, mode="a", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["Model", "Block_Index", "Result", "Time_Seconds"])

    for model_id in model_ids:
        print(f"\n--- Loading model: {model_id} ---")
    
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=dtype
        )

        for i, block_text in enumerate(biosample_blocks):
            messages = [
                {
                    "role": "user",
                    "content": f"""Extract the most accurate geographic location information from this XML block (latitude and longitude or region or country for instance). 
Return ONLY the location string with no additional text. If the location is not possible write "NA"
XML Block:
{block_text}"""
                }
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            start_time = time.time()
            output = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            end_time = time.time()

            input_len = inputs.input_ids.shape[1]
            gen_tokens = output[0][input_len:]
            result = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            time_taken = end_time - start_time

            print(f"\n=== Model: {model_id} | Block {i} ===")
            print(f"Time: {time_taken:.2f}s | Result: {result}")

            writer.writerow([model_id, i, result, f"{time_taken:.4f}"])
