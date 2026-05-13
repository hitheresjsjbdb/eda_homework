from __future__ import annotations

import json
from pathlib import Path

import torch


def load_split(path: Path, split_name: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data[split_name])


def read_ref_qor(case_dir: Path) -> dict[str, int] | None:
    ref = case_dir / "ref_qor.txt"
    if not ref.is_file():
        return None
    out = {}
    for line in ref.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = int(value.strip())
    return out if out else None


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def progress_iter(iterable, **kwargs):
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable
