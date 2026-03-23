import json
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATASET_NAME = "redcathode/thingiverse-openscad"  # Hugging Face dataset
MODEL_NAME = "unsloth/Qwen2.5-Coder-1.5B"  # Highly capable, fits in 4GB VRAM
MAX_SEQ_LENGTH = 1024  # Limit context to save VRAM
OUTPUT_DIR = "openscad_lora_model"

# ── 2. Load and Prepare Dataset ───────────────────────────────────────────────
print("Loading dataset from Hugging Face...")
dataset = load_dataset(DATASET_NAME)

# Get the split (usually 'train')
if "train" in dataset:
    dataset = dataset["train"]
else:
    dataset = dataset[list(dataset.keys())[0]]

print(f"Loaded dataset with {len(dataset)} examples")

# Define the Prompt Format (Alpaca Style)
alpaca_prompt = """Below is a description of a 3D object. Write valid OpenSCAD code to generate it.

### Description:
{}

### OpenSCAD Code:
{}"""

EOS_TOKEN = "<|endoftext|>"  # End of sequence token for Qwen

# Determine field names based on dataset structure
# Try common field names for prompt/description and code
def get_field_names(dataset):
    """Determine the field names for text and code in the dataset."""
    features = dataset.features
    
    # Look for description/prompt fields
    text_fields = ["description", "prompt", "text", "input"]
    code_fields = ["code", "output", "openscad", "script"]
    
    text_field = None
    for field in text_fields:
        if field in features:
            text_field = field
            break
    
    code_field = None
    for field in code_fields:
        if field in features:
            code_field = field
            break
    
    return text_field, code_field

text_field, code_field = get_field_names(dataset)
print(f"Using text field: {text_field}, code field: {code_field}")

def formatting_prompts_func(examples):
    prompts = examples[text_field]
    codes = examples[code_field]
    texts = []
    for prompt, code in zip(prompts, codes):
        text = alpaca_prompt.format(prompt, code) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ── 3. Load Model (4-bit Quantized) ───────────────────────────────────────────
print("Loading Model in 4-bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,           # Auto-detects (FP16/BF16)
    load_in_4bit = True,    # CRUCIAL FOR 4GB VRAM!
)

# ── 4. Setup LoRA (Low-Rank Adaptation) ───────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: 16 is a good balance for code generation
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth optimizes dropout = 0
    bias = "none",
    use_gradient_checkpointing = "unsloth", # CRUCIAL: Saves massive VRAM
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ── 5. Setup Trainer ──────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # Can make training faster for short sequences
    args = TrainingArguments(
        per_device_train_batch_size = 1,      # VRAM savior
        gradient_accumulation_steps = 4,      # Simulates batch_size of 4
        warmup_steps = 10,
        max_steps = 200,                      # Adjust based on dataset size (or use num_train_epochs = 1)
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),   # Ampere (RTX 3050) supports bfloat16
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",                 # Saves optimizer VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to="none",                     # Disable wandb/tensorboard for clean terminal
    ),
)

# ── 6. Start Training! ────────────────────────────────────────────────────────
print("\n🚀 Starting Fine-Tuning! Monitor your VRAM in Task Manager/nvtop.\n")
trainer_stats = trainer.train()

# ── 7. Save the Tuned Model ───────────────────────────────────────────────────
print(f"Saving LoRA adapters to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR) # Local saving
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete!")