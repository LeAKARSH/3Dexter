from unsloth import FastLanguageModel
import torch

MODEL_DIR = "openscad_lora_model_3b_2"
MAX_SEQ_LENGTH = 1024

alpaca_prompt = """Below is a description of a 3D object. Write valid OpenSCAD code to generate it.

### Description:
{}

### OpenSCAD Code:
"""

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# Test prompt
test_prompt = "A simple red cube with side length 20mm"

print(f"\n{'='*60}")
print(f"TESTING: {test_prompt}")
print('='*60)

# Generate
input_text = alpaca_prompt.format(test_prompt)
inputs = tokenizer([input_text], return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)

# Extract ONLY the first code block before any repetition
# Stop at the first occurrence of prompt patterns
stop_patterns = ["### Description:", "### Human:", "Human:", "Below is a description"]
for pattern in stop_patterns:
    if pattern in generated_code:
        generated_code = generated_code.split(pattern)[0].strip()
        break

# If there's still a "### OpenSCAD Code:" header, skip it
if "### OpenSCAD Code:" in generated_code:
    generated_code = generated_code.split("### OpenSCAD Code:")[-1].strip()

print("\nGENERATED CODE:")
print("="*60)
print(generated_code)
print("="*60)

# Save to file
with open("test_cube.scad", "w") as f:
    f.write(generated_code)
print("\nSaved to test_cube.scad")

# Check braces
open_braces = generated_code.count("{")
close_braces = generated_code.count("}")
print(f"\nBrace count: {open_braces} open, {close_braces} close")
if open_braces == close_braces:
    print("✅ Braces balanced!")
else:
    print(f"❌ Unbalanced: {abs(open_braces - close_braces)} difference")
