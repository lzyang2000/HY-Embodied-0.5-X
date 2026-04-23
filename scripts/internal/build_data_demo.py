"""Internal build script — collect accessible samples into data_examples/data_demo.jsonl.

This script is only used when packaging the repo for release. It:
    1. reads a fixed set of source JSONL files under data_examples/ (expected
       to be prepared separately from our internal training corpus),
    2. picks up to N samples whose referenced images are actually readable on
       this machine,
    3. copies those images into data_examples/images/<category>/,
    4. rewrites the "image" fields to repo-relative paths,
    5. writes the merged result to data_examples/data_demo.jsonl.

Not imported by the package — this is a one-off developer tool.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "data_examples"
OUT_JSONL = SRC_DIR / "data_demo.jsonl"
IMG_DIR = SRC_DIR / "images"

SPECS = [
    # (category, source_filename, max_samples, max_images_per_sample)
    ("planning",     "planning.jsonl",            2, 0),   # pure text
    ("trajectory",   "trajectory.jsonl",          2, 1),
    ("affordance",   "affordance.jsonl",          2, 4),   # cap frames to avoid explosion
    ("refspatial",   "refspatial.jsonl",          2, 1),
    ("robot_traj",   "robot_trajectory_qa.jsonl", 2, 1),
    # demo_self_contained already uses assets/ images — include as-is
]


def _image_paths(record: dict) -> list[str]:
    paths: list[str] = []
    for m in record.get("messages", []):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in ("image", "video", "image_url"):
                if key in item and isinstance(item[key], str):
                    paths.append(item[key])
    return paths


def _all_accessible(record: dict) -> bool:
    paths = _image_paths(record)
    return all(Path(p).exists() for p in paths)


def _cap_images(record: dict, max_images: int) -> dict | None:
    """Keep at most `max_images` image items (drop the rest). 0 = keep zero.

    Returns None if the record has fewer than 1 text token after capping (safety).
    """
    new_messages = []
    kept_images = 0
    for m in record["messages"]:
        content = m.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and (
                    item.get("type") in ("image", "video")
                    or any(k in item for k in ("image", "video", "image_url"))
                ):
                    if max_images == 0:
                        continue
                    if kept_images >= max_images:
                        continue
                    kept_images += 1
                new_content.append(item)
            new_messages.append({**m, "content": new_content})
        else:
            new_messages.append(m)
    return {**record, "messages": new_messages}


def _rewrite_image_paths(record: dict, category: str, sample_idx: int) -> dict:
    """Copy images to data_examples/images/<category>/ and rewrite paths to repo-relative."""
    dst_subdir = IMG_DIR / category
    dst_subdir.mkdir(parents=True, exist_ok=True)

    img_idx = 0
    for m in record["messages"]:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            for key in ("image", "video", "image_url"):
                if key in item and isinstance(item[key], str):
                    src = Path(item[key])
                    if not src.exists():
                        continue
                    suffix = src.suffix or ".jpg"
                    dst_name = f"{category}_{sample_idx:02d}_{img_idx:02d}{suffix}"
                    img_idx += 1
                    dst = dst_subdir / dst_name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    item[key] = str(dst.relative_to(REPO))
    return record


def collect() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = []

    # --- 1. Always include the self-contained demo (uses assets/) ---
    sc_path = SRC_DIR / "demo_self_contained.jsonl"
    if sc_path.exists():
        with sc_path.open() as f:
            for line in f:
                if line.strip():
                    out_lines.append(line.rstrip("\n"))

    # --- 2. Iterate per-category sources ---
    for category, filename, max_samples, max_images in SPECS:
        src = SRC_DIR / filename
        if not src.exists():
            print(f"  SKIP {filename}: not found")
            continue

        kept = 0
        with src.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # First cap images to bound the demo footprint.
                capped = _cap_images(rec, max_images)
                if capped is None:
                    continue
                if not _all_accessible(capped):
                    continue
                rewritten = _rewrite_image_paths(capped, category, kept)
                rewritten.setdefault("_category", category)
                out_lines.append(json.dumps(rewritten, ensure_ascii=False))
                kept += 1
                if kept >= max_samples:
                    break
        print(f"  {category:12s} <- {filename}: kept {kept}/{max_samples}")

    OUT_JSONL.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_JSONL} ({len(out_lines)} samples)")


if __name__ == "__main__":
    collect()
