"""
Step 1 — Cap3D Dataset Download & Parametric Caption Filter
============================================================
Downloads Cap3D captions from Hugging Face and filters for
descriptions that are translatable to OpenSCAD primitives,
transformations, and boolean operations.
"""

import os
import re
import json
import csv
import requests
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN         = ""          # optional; Cap3D is public
OUTPUT_DIR       = Path("cap3d_data")
OUTPUT_FILE      = OUTPUT_DIR / "parametric_captions.json"
MAX_EXAMPLES     = 100_000     # how many raw captions to scan
MIN_LEN          = 20
MAX_LEN          = 300
SAVE_EVERY       = 2_000

# Cap3D CSV hosted directly on HF datasets (no streaming needed)
CAP3D_CSV_URL = (
    "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/"
    "Cap3D_automated_Objaverse_full.csv"
)

# ── Parametric filter vocabulary ──────────────────────────────────────────────
GEOMETRY_TERMS = {
    # primitives
    "cube", "box", "cuboid", "rectangular prism",
    "cylinder", "cylindrical", "tube", "pipe", "rod", "shaft",
    "sphere", "spherical", "ball",
    "cone", "conical", "frustum",
    "torus", "ring", "donut",
    "prism", "pyramid", "wedge", "polyhedron",
    # structural / mechanical
    "bracket", "mount", "flange", "plate", "panel", "slab", "block",
    "gear", "screw", "bolt", "nut", "washer", "thread", "helix",
    "slot", "groove", "chamfer", "fillet", "bevel", "notch",
    "hole", "bore", "pocket", "extrusion", "recess", "cutout",
    "rail", "channel", "rib", "fin", "tab", "lip",
    # transformations / spatial
    "rotate", "rotated", "rotation",
    "translate", "translated", "offset",
    "mirror", "symmetric", "symmetrical",
    "scale", "scaled",
    "linear pattern", "circular pattern", "array",
    # boolean ops
    "union", "difference", "intersection",
    "subtract", "subtracted", "combined", "merged",
    "hollow", "shell", "wall",
    # dimensions / measurements
    r"\d+\s*mm", r"\d+\s*cm", r"\d+\s*inch",
    "diameter", "radius", "height", "width", "depth",
    "length", "thickness", "angle", "degree",
    "dimension", "measurement", "parametric",
}

# Compile once: literal terms as \b-bounded, regex terms as-is
_LITERAL_RE = re.compile(
    r"|".join(
        r"\b" + re.escape(t) + r"\b"
        if not any(c in t for c in r"\.+*?[](){}^$|\\")
        else t
        for t in GEOMETRY_TERMS
    ),
    re.IGNORECASE,
)

# Terms that indicate non-parametric organic/scene content
REJECT_TERMS = re.compile(
    r"\b(person|human|face|hair|animal|tree|plant|grass|sky|"
    r"cartoon|character|monster|creature|landscape|building facade|"
    r"texture|material|cloth|fabric|fur|skin)\b",
    re.IGNORECASE,
)

SCORE_THRESHOLD = 2   # minimum number of distinct parametric hits


def parametric_score(text: str) -> int:
    """Return count of distinct parametric keyword matches."""
    return len(set(m.group(0).lower() for m in _LITERAL_RE.finditer(text)))


def is_parametric(text: str) -> bool:
    if len(text) < MIN_LEN or len(text) > MAX_LEN:
        return False
    if REJECT_TERMS.search(text):
        return False
    return parametric_score(text) >= SCORE_THRESHOLD


# ── Download ──────────────────────────────────────────────────────────────────

def download_cap3d_csv(dest: Path) -> Path:
    csv_path = OUTPUT_DIR / "Cap3D_full.csv"
    if csv_path.exists():
        print(f"✅ CSV already cached at {csv_path}")
        return csv_path

    print(f"📥 Downloading Cap3D CSV …")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    resp = requests.get(CAP3D_CSV_URL, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Cap3D CSV"
    ) as bar:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"✅ Saved to {csv_path}")
    return csv_path


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = download_cap3d_csv(OUTPUT_DIR)

    filtered: list[dict] = []
    seen_uids: set[str] = set()

    print(f"🔍 Filtering captions (target: {MAX_EXAMPLES:,} scanned) …")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(tqdm(reader, desc="Scanning")):
            if i >= MAX_EXAMPLES:
                break
            if len(row) < 2:
                continue

            uid, caption = row[0].strip(), row[1].strip()
            if uid in seen_uids:
                continue

            if is_parametric(caption):
                seen_uids.add(uid)
                filtered.append({
                    "uid":   uid,
                    "text":  caption,
                    "score": parametric_score(caption),
                })

            if len(filtered) % SAVE_EVERY == 0 and len(filtered) > 0:
                _save(filtered)
                print(f"  💾 Checkpoint: {len(filtered):,} captions retained")

    _save(filtered)
    print(f"\n✅ Done! {len(filtered):,} parametric captions → {OUTPUT_FILE}")
    if filtered:
        print("\n📝 Sample:")
        for item in sorted(filtered, key=lambda x: -x["score"])[:5]:
            print(f"   [{item['score']}] {item['text'][:120]}")

    return filtered


def _save(data):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    run()
