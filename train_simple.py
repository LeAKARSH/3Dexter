import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATASET_PATH = "generated_code_cap3d.jsonl"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"  # Using official model name
MAX_SEQ_LENGTH = 1024
OUTPUT_DIR = "openscad_lora_model"

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

print(f"Loaded {len(data)} examples")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)

# Define the Prompt Format (Alpaca Style)
alpaca_prompt = """Below is a description of a 3D object. Write valid OpenSCAD code to generate it.

### Description:
{}

### OpenSCAD Code:
{}"""

# ── 3. Load Model and Tokenizer ───────────────────────────────────────────────
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# ── 4. Setup LoRA ─────────────────────────────────────────────────────────────
print("Setting up LoRA...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ── 5. Tokenize Dataset ───────────────────────────────────────────────────────
def tokenize_function(examples):
    prompts = examples["prompt"]
    codes = examples["code"]
    texts = [alpaca_prompt.format(p, c) for p, c in zip(prompts, codes)]
    
    result = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
        return_tensors=None
    )
    result["labels"] = result["input_ids"].copy()
    return result

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)

# ── 6. Setup Trainer ──────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    warmup_steps=10,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# ── 7. Start Training! ────────────────────────────────────────────────────────
print("\n🚀 Starting Fine-Tuning!\n")
trainer.train()

# ── 8. Save the Tuned Model ───────────────────────────────────────────────────
print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete!")
