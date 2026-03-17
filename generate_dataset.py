import google.genai as genai
import json
import os
import time
from tqdm import tqdm

# Configure API key - REPLACE WITH YOUR ACTUAL API KEY
API_KEY = ""  # <-- Replace this with your actual Gemini API key

# Initialize the client
client = genai.Client(api_key=API_KEY)

# Load your filtered captions - with UTF-8 encoding
with open("objaverse_data/filtered_captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

# We'll work with a subset or all
captions = captions[:5000]   # or however many you want

output_file = "generated_code_gemini.jsonl"
start_idx = 0

# Resume if file already exists
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        start_idx = len(lines)
    print(f"Resuming from index {start_idx}")

# Define prompt template
def build_prompt(caption):
    return f"""Generate valid OpenSCAD code for the following object. 
Only output the code, no explanations.

Object: {caption}

OpenSCAD code:"""

# Optional: Test available models (uncomment to see what's available)
# print("Available models:")
# for model in client.models.list():
#     print(f"- {model.name}")

# Open output file in append mode with UTF-8 encoding
with open(output_file, "a", encoding="utf-8") as f_out:
    for i, item in enumerate(tqdm(captions[start_idx:], desc="Generating", initial=start_idx, total=len(captions))):
        idx = start_idx + i
        caption = item['text']
        prompt = build_prompt(caption)

        try:
            # Fixed model name format - use full path format
            response = client.models.generate_content(
                model='models/gemma-3-27B',  # Changed from 'gemini-1.5-flash' to 'models/gemini-1.5-flash'
                contents=prompt
            )
            code = response.text.strip()
            
            # Optional: remove markdown code fences if present
            if code.startswith("```") and code.endswith("```"):
                code = code.split("```")[1].strip()
                if code.startswith("openscad"):
                    code = code[8:].strip()
                    
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

        # Rate limit: 60 requests per minute = 1 request per second
        time.sleep(1)

print("Done!")