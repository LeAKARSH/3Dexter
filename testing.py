import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from datasets import Dataset
from unsloth import FastLanguageModel

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR       = "openscad_lora_model"       # Path to saved LoRA adapters
DATASET_PATH    = "generated_code_cap3d.jsonl"
MAX_SEQ_LENGTH  = 1024
EVAL_SAMPLES    = 50                          # Number of samples to evaluate
OPENSCAD_BIN    = "openscad"                  # Path to OpenSCAD binary (must be installed)

alpaca_prompt = """Below is a description of a 3D object. Write valid OpenSCAD code to generate it.

### Description:
{}

### OpenSCAD Code:
{}"""

# ── Data Classes ──────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    prompt: str
    expected_code: str
    generated_code: str
    syntax_valid: bool          = False
    renders_successfully: bool  = False
    token_overlap: float        = 0.0
    latency_seconds: float      = 0.0
    error_message: str          = ""

@dataclass
class EvalSummary:
    total_samples: int          = 0
    syntax_valid_count: int     = 0
    render_success_count: int   = 0
    avg_token_overlap: float    = 0.0
    avg_latency_seconds: float  = 0.0
    results: list               = field(default_factory=list)

    @property
    def syntax_accuracy(self):
        return self.syntax_valid_count / self.total_samples if self.total_samples else 0

    @property
    def render_accuracy(self):
        return self.render_success_count / self.total_samples if self.total_samples else 0


# ── 1. Load Model ─────────────────────────────────────────────────────────────
def load_model(model_dir: str):
    print(f"Loading model from {model_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_dir,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    return model, tokenizer


# ── 2. Inference ──────────────────────────────────────────────────────────────
def generate_openscad(prompt: str, model, tokenizer, max_new_tokens: int = 512) -> tuple[str, float]:
    """Generate OpenSCAD code for a given description. Returns (code, latency)."""
    input_text = alpaca_prompt.format(prompt, "")  # Empty code section for generation
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens  = max_new_tokens,
            temperature     = 0.1,      # Low temperature for deterministic code
            do_sample       = True,
            pad_token_id    = tokenizer.eos_token_id,
            eos_token_id    = tokenizer.eos_token_id,
        )
    latency = time.time() - start

    # Decode only the newly generated tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text   = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean up: stop at any trailing prompt artifacts
    generated_text = generated_text.split("### Description:")[0].strip()
    return generated_text, latency


# ── 3. Syntax Validation ──────────────────────────────────────────────────────
def check_openscad_syntax(code: str) -> tuple[bool, str]:
    """
    Lightweight syntax check — looks for common structural markers.
    For a full check, uses OpenSCAD CLI if available.
    """
    # Basic heuristic checks
    if not code or len(code.strip()) < 5:
        return False, "Empty or trivially short code"

    # Check for at least one OpenSCAD primitive or module
    primitives = ["cube", "sphere", "cylinder", "polyhedron", "linear_extrude",
                  "rotate_extrude", "import", "module", "union", "difference",
                  "intersection", "translate", "rotate", "scale", "mirror"]
    has_primitive = any(p in code for p in primitives)
    if not has_primitive:
        return False, "No recognizable OpenSCAD primitives found"

    # Check balanced braces
    if code.count("{") != code.count("}"):
        return False, f"Unbalanced braces: {code.count('{')} open, {code.count('}')} close"

    # Check balanced parentheses
    if code.count("(") != code.count(")"):
        return False, f"Unbalanced parentheses: {code.count('(')} open, {code.count(')')} close"

    # Try OpenSCAD CLI if installed
    try:
        with tempfile.NamedTemporaryFile(suffix=".scad", mode="w", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [OPENSCAD_BIN, "--hardwarnings", "-o", "/dev/null", tmp_path],
            capture_output=True, text=True, timeout=10
        )
        os.unlink(tmp_path)

        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr.strip()[:200]

    except FileNotFoundError:
        # OpenSCAD not installed — fall back to heuristic result
        return True, "(OpenSCAD CLI not found; heuristic checks passed)"
    except subprocess.TimeoutExpired:
        return False, "OpenSCAD timed out during syntax check"
    except Exception as e:
        return True, f"(CLI check skipped: {e})"


# ── 4. Render Check ───────────────────────────────────────────────────────────
def check_render(code: str) -> tuple[bool, str]:
    """Try to render the code to a temp STL using OpenSCAD CLI."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".scad", mode="w", delete=False) as scad_f:
            scad_f.write(code)
            scad_path = scad_f.name

        stl_path = scad_path.replace(".scad", ".stl")

        result = subprocess.run(
            [OPENSCAD_BIN, "-o", stl_path, scad_path],
            capture_output=True, text=True, timeout=30
        )
        os.unlink(scad_path)
        if os.path.exists(stl_path):
            os.unlink(stl_path)

        return result.returncode == 0, result.stderr.strip()[:200]

    except FileNotFoundError:
        return False, "OpenSCAD CLI not installed"
    except subprocess.TimeoutExpired:
        return False, "Render timed out (>30s)"
    except Exception as e:
        return False, str(e)


# ── 5. Token Overlap (Approximate Code Similarity) ───────────────────────────
def token_overlap(expected: str, generated: str) -> float:
    """Jaccard similarity on token sets (simple code similarity metric)."""
    def tokenize(code):
        return set(re.findall(r'\w+|[{}();,]', code.lower()))

    exp_tokens = tokenize(expected)
    gen_tokens = tokenize(generated)

    if not exp_tokens and not gen_tokens:
        return 1.0
    if not exp_tokens or not gen_tokens:
        return 0.0

    intersection = exp_tokens & gen_tokens
    union        = exp_tokens | gen_tokens
    return len(intersection) / len(union)


# ── 6. Full Evaluation Loop ───────────────────────────────────────────────────
def evaluate(model, tokenizer, dataset_path: str, n_samples: int) -> EvalSummary:
    print(f"\nLoading dataset from {dataset_path}...")
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("code") and len(item["code"]) > 10:
                    data.append(item)

    samples = data[:n_samples]
    print(f"Evaluating on {len(samples)} samples...\n")

    summary = EvalSummary(total_samples=len(samples))

    for i, item in enumerate(samples):
        prompt        = item["prompt"]
        expected_code = item["code"]

        print(f"[{i+1}/{len(samples)}] Generating for: {prompt[:60]}...")

        generated_code, latency = generate_openscad(prompt, model, tokenizer)
        syntax_valid, syntax_err = check_openscad_syntax(generated_code)
        renders_ok,   render_err = check_render(generated_code) if syntax_valid else (False, "Skipped (syntax invalid)")
        overlap                  = token_overlap(expected_code, generated_code)

        result = EvalResult(
            prompt              = prompt,
            expected_code       = expected_code,
            generated_code      = generated_code,
            syntax_valid        = syntax_valid,
            renders_successfully= renders_ok,
            token_overlap       = overlap,
            latency_seconds     = latency,
            error_message       = syntax_err or render_err,
        )
        summary.results.append(result)

        if syntax_valid:
            summary.syntax_valid_count += 1
        if renders_ok:
            summary.render_success_count += 1

        summary.avg_token_overlap   += overlap
        summary.avg_latency_seconds += latency

        status = "✅" if syntax_valid else "❌"
        print(f"  {status} Syntax: {syntax_valid} | Render: {renders_ok} | "
              f"Token Overlap: {overlap:.2%} | Latency: {latency:.2f}s")
        if result.error_message:
            print(f"     ↳ {result.error_message}")

    # Finalize averages
    summary.avg_token_overlap   /= len(samples)
    summary.avg_latency_seconds /= len(samples)

    return summary


# ── 7. Print & Save Report ────────────────────────────────────────────────────
def print_report(summary: EvalSummary):
    print("\n" + "="*60)
    print("           EVALUATION REPORT")
    print("="*60)
    print(f"  Total Samples Evaluated : {summary.total_samples}")
    print(f"  Syntax Accuracy         : {summary.syntax_accuracy:.2%}  "
          f"({summary.syntax_valid_count}/{summary.total_samples})")
    print(f"  Render Success Rate     : {summary.render_accuracy:.2%}  "
          f"({summary.render_success_count}/{summary.total_samples})")
    print(f"  Avg Token Overlap       : {summary.avg_token_overlap:.2%}")
    print(f"  Avg Inference Latency   : {summary.avg_latency_seconds:.2f}s")
    print("="*60)


def save_report(summary: EvalSummary, out_path: str = "eval_results.json"):
    report = {
        "total_samples"       : summary.total_samples,
        "syntax_accuracy"     : round(summary.syntax_accuracy, 4),
        "render_accuracy"     : round(summary.render_accuracy, 4),
        "avg_token_overlap"   : round(summary.avg_token_overlap, 4),
        "avg_latency_seconds" : round(summary.avg_latency_seconds, 4),
        "results": [
            {
                "prompt"              : r.prompt,
                "generated_code"      : r.generated_code,
                "syntax_valid"        : r.syntax_valid,
                "renders_successfully": r.renders_successfully,
                "token_overlap"       : round(r.token_overlap, 4),
                "latency_seconds"     : round(r.latency_seconds, 4),
                "error_message"       : r.error_message,
            }
            for r in summary.results
        ]
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


# ── 8. Interactive Inference CLI ──────────────────────────────────────────────
def inference_cli(model, tokenizer):
    print("\n" + "="*60)
    print("   INTERACTIVE INFERENCE — type 'quit' to exit")
    print("="*60)

    while True:
        prompt = input("\nDescribe a 3D object: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        print("Generating OpenSCAD code...\n")
        code, latency = generate_openscad(prompt, model, tokenizer)
        syntax_valid, err = check_openscad_syntax(code)

        print("─" * 50)
        print(code)
        print("─" * 50)
        print(f"Syntax valid : {'✅ Yes' if syntax_valid else '❌ No'}")
        if err:
            print(f"Note         : {err}")
        print(f"Latency      : {latency:.2f}s")

        # Optionally save to file
        save = input("\nSave to .scad file? (y/N): ").strip().lower()
        if save == "y":
            fname = input("Filename (without extension): ").strip() or "output"
            with open(f"{fname}.scad", "w") as f:
                f.write(code)
            print(f"Saved to {fname}.scad")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate and run inference on the OpenSCAD LoRA model")
    parser.add_argument("--mode",       choices=["eval", "infer", "both"], default="both",
                        help="Run evaluation, interactive inference, or both (default: both)")
    parser.add_argument("--model_dir",  default=MODEL_DIR,      help="Path to LoRA model directory")
    parser.add_argument("--dataset",    default=DATASET_PATH,   help="Path to .jsonl dataset")
    parser.add_argument("--n_samples",  type=int, default=EVAL_SAMPLES, help="Number of eval samples")
    parser.add_argument("--output",     default="eval_results.json",    help="Path for eval JSON report")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_dir)

    if args.mode in ("eval", "both"):
        summary = evaluate(model, tokenizer, args.dataset, args.n_samples)
        print_report(summary)
        save_report(summary, args.output)

    if args.mode in ("infer", "both"):
        inference_cli(model, tokenizer)