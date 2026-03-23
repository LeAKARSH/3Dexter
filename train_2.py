import json
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATASET_PATH = "generated_code_cap3d.jsonl"
#MODEL_NAME = "unsloth/Qwen2.5-Coder-1.5B" # Highly capable, fits in 4GB VRAM
MODEL_NAME = "unsloth/Qwen2.5-Coder-3B"   # Best balance for 6GB VRAM
#MODEL_NAME = "unsloth/Qwen2.5-Coder-7B"  # Too large for 6GB VRAM
MAX_SEQ_LENGTH = 1024                     # 3B can handle full context easily
OUTPUT_DIR = "openscad_lora_model_3b_2"     # Save to different folder

# ── 2. Load and Prepare Dataset ───────────────────────────────────────────────
print("Loading dataset...")
data = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            # Skip empty code generations or errors
            if item.get("code") and len(item["code"]) > 10:
                data.append(item)

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)

# Define the Prompt Format (Alpaca Style)
alpaca_prompt = """Below is a description of a 3D object. Write valid OpenSCAD code to generate it.

### Description:
{}

### OpenSCAD Code:
{}"""

EOS_TOKEN = "<|endoftext|>" # End of sequence token for Qwen

def formatting_prompts_func(examples):
    prompts = examples["prompt"]
    codes = examples["code"]
    texts = []
    for prompt, code in zip(prompts, codes):
        text = alpaca_prompt.format(prompt, code) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

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
        gradient_accumulation_steps = 4,      # Good for 6GB VRAM with 7B model
        warmup_steps = 10,
        num_train_epochs = 5,                 # Increased from 3 for better convergence
        #max_steps = 200,                   # Or limit to 200 steps for quick testing
        learning_rate = 1e-4,                 # Reduced from 2e-4 for 3B stability
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