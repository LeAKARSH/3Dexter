from groq import Groq
import json
import os
import time
from tqdm import tqdm

# ==============================
# Configure Groq API Key
# ==============================
GROQ_API_KEY = ""  # <-- Replace with your actual Groq key

client = Groq(api_key=GROQ_API_KEY)

# ==============================
# Load captions
# ==============================
with open("objaverse_data/filtered_captions.json", "r", encoding="utf-8") as f:
    captions = json.load(f)

captions = captions[:5000]   # Limit if needed

output_file = "generated_code_groq.jsonl"
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
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # You can change to a code-focused model if available
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

        # Groq is fast, but add small delay if needed
        time.sleep(0.5)

print("Done!")