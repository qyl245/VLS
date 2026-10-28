"""Resolve QEVD-FIT data paths across workstations.

Resolution order:
1. ``VLS_DATA_ROOT`` environment variable.
2. ``<repo>/rawdata/QEVD-FIT-COACH`` (current Windows checkout).
3. ``<repo>/dataset/QEVD-FIT/QEVD-FIT-COACH``.
4. ``<repo>/../dataset/QEVD-FIT/QEVD-FIT-COACH`` (original workstation layout).
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_VAR = "VLS_DATA_ROOT"
DATASET_NAME = "QEVD-FIT-COACH"


def data_root_candidates() -> list[Path]:
    candidates = []
    env_value = os.environ.get(ENV_VAR)
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(
        [
            ROOT / "rawdata" / DATASET_NAME,
            ROOT / "dataset" / "QEVD-FIT" / DATASET_NAME,
            ROOT.parent / "dataset" / "QEVD-FIT" / DATASET_NAME,
        ]
    )
    return candidates


def resolve_data_root() -> Path:
    for candidate in data_root_candidates():
        if candidate.is_dir():
            return candidate.resolve()
    return (ROOT / "rawdata" / DATASET_NAME).resolve()


DATA_ROOT = resolve_data_root()


def resolve_data_path(value: str | Path) -> Path:
    """Return an existing local path, remapping legacy dataset roots if needed."""
    path = Path(value)
    if path.exists():
        return path
    parts = path.parts
    for index, part in enumerate(parts):
        if part.upper() == DATASET_NAME:
            return DATA_ROOT.joinpath(*parts[index + 1 :])
    candidate = DATA_ROOT / path.name
    if candidate.exists():
        return candidate
    return path
