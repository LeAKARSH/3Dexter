import os
import json
from tqdm import tqdm
from datasets import load_dataset

# Disable symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Hard‑code your Hugging Face token (REPLACE WITH A NEW TOKEN!)
HF_TOKEN = ""

def download_and_filter_objaverse(
    output_dir="objaverse_data",
    max_captions=20000,
    min_len=15,
    max_len=250,
    save_every=1000
):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "filtered_captions.json")

    # Try "caption" config first
    try:
        print("📥 Attempting to load 'caption' config...")
        ds = load_dataset(
            "allenai/objaverse-xl",
            "caption",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        config_used = "caption"
    except Exception as e:
        print(f"⚠️ 'caption' config failed: {e}")
        print("📥 Falling back to 'default' config...")
        ds = load_dataset(
            "allenai/objaverse-xl",
            "default",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        config_used = "default"

    # Peek at first example
    it = iter(ds)
    first = next(it)
    print(f"First example type: {type(first)}")

    # Determine format and how to extract text
    if isinstance(first, dict):
        print("Detected dictionary format. Keys in first example:")
        for k, v in first.items():
            # Show key, type, and a preview of value
            print(f"  {k}: type={type(v).__name__}, preview={str(v)[:100]}")
        
        # Try to find a suitable text key
        text_key = None
        # Preferred keys in order
        preferred = ['caption', 'text', 'description', 'name']
        for key in preferred:
            if key in first and isinstance(first[key], str) and len(first[key]) > 0:
                text_key = key
                print(f"✅ Using key '{text_key}' for captions.")
                break
        
        # If none of the preferred keys work, look for any string field
        if not text_key:
            print("⚠️ Preferred keys not found. Searching for any string field...")
            for key, value in first.items():
                if isinstance(value, str) and len(value) >= min_len:
                    text_key = key
                    print(f"✅ Found string field '{key}' (length {len(value)}) – using it.")
                    break
        
        if not text_key:
            raise ValueError("No suitable string field found in dictionary examples.")
        
        # Rebuild iterator (since we consumed one)
        ds = load_dataset(
            "allenai/objaverse-xl",
            config_used,
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        extract_func = lambda ex: ex.get(text_key, "")
        
    elif isinstance(first, str):
        print("Detected string format. Using each example directly as caption.")
        # If first string is a split name (like 'train'), we might need to skip it
        # But for now, assume all strings are captions.
        ds = load_dataset(
            "allenai/objaverse-xl",
            config_used,
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        extract_func = lambda ex: ex
    else:
        raise TypeError(f"Unexpected example type: {type(first)}")

    # Keywords for filtering parametric objects
    parametric_keywords = [
        'cube', 'cylinder', 'sphere', 'box', 'pipe', 'tube', 'gear', 'screw',
        'bolt', 'nut', 'washer', 'bracket', 'mount', 'holder', 'container',
        'cup', 'mug', 'bowl', 'plate', 'disk', 'ring', 'spacer',
        'rectangular', 'square', 'circular', 'cylindrical', 'prism',
        'hole', 'slot', 'groove', 'thread', 'extrusion', 'rotation',
        'diameter', 'radius', 'height', 'width', 'depth', 'length',
        'mm', 'cm', 'inch', 'parametric', 'dimension', 'measurement'
    ]

    filtered = []
    try:
        for i, example in enumerate(tqdm(ds, desc="Processing", total=max_captions)):
            if i >= max_captions:
                break

            caption = extract_func(example)
            if not isinstance(caption, str) or len(caption) < min_len:
                continue

            text_lower = caption.lower()
            if any(kw in text_lower for kw in parametric_keywords):
                if min_len <= len(caption) <= max_len:
                    # Generate a unique ID
                    uid = f"obj_{i}"
                    if isinstance(example, dict):
                        uid = example.get('uid', example.get('id', uid))
                    filtered.append({'uid': uid, 'text': caption})

            # Periodic save
            if len(filtered) % save_every == 0 and len(filtered) > 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(filtered, f, indent=2, ensure_ascii=False)
                print(f"\n💾 Saved checkpoint with {len(filtered)} captions.")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user – saving current progress.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Collected {len(filtered)} parametric-friendly captions.")
    if filtered:
        print("\n📝 Sample:")
        for item in filtered[:5]:
            print(f"   - {item['text'][:100]}...")
    return filtered

if __name__ == "__main__":
    download_and_filter_objaverse(max_captions=20000)