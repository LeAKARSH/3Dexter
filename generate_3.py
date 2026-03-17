import json
import os
import time
from tqdm import tqdm
from openai import OpenAI  # requires: pip install openai

# ==============================
# Configure OpenRouter API Key
# ==============================
OPENROUTER_API_KEY = ""  # <-- Replace with your actual OpenRouter key

# Initialize the OpenAI client with OpenRouter's base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ==============================
# Load captions
# ==============================
with open("objaverse_data/filtered_captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

captions = captions[:5000]   

output_file = "generated_code_openrouter.jsonl"
start_idx = 0

# Resume support
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        start_idx = len(lines)
    print(f"Resuming from index {start_idx}")

# ==============================
# Prompt Template
# ==============================
def build_prompt(caption):
    return f"""Generate valid OpenSCAD code for the following object.
Only output the code, no explanations.

Object: {caption}

OpenSCAD code:"""

# ==============================
# Generation Loop
# ==============================
with open(output_file, "a", encoding="utf-8") as f_out:
    for i, item in enumerate(tqdm(captions[start_idx:], 
                                  desc="Generating", 
                                  initial=start_idx, 
                                  total=len(captions))):
        idx = start_idx + i
        caption = item['text']
        prompt = build_prompt(caption)

        try:
            # Use a model available on OpenRouter that is good for code generation.
            # Examples: "openai/gpt-4o", "meta-llama/llama-3.3-70b-instruct", "mistralai/mixtral-8x7b-instruct"
            response = client.chat.completions.create(
                model="google/gemini-3.1-flash-lite-preview",  # adjust as needed
                messages=[
                    {"role": "system", "content": "You are an expert OpenSCAD code generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
            )

            code = response.choices[0].message.content.strip()

            # Remove markdown fences if present
            if code.startswith("```"):
                code = code.replace("```openscad", "").replace("```", "").strip()

        except Exception as e:
            code = ""
            print(f"Error at {idx}: {e}")

        record = {
            'uid': item.get('uid', f'cap_{idx}'),
            'prompt': caption,
            'code': code
        }

        f_out.write(json.dumps(record) + '\n')
        f_out.flush()

        # Small delay to avoid hitting rate limits
        time.sleep(0.5)

print("Done!")