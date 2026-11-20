"""Resolve immutable run-record paths after the 2026-11-17 directory migration.

Only path values are relocated. Original JSON/checkpoints remain unchanged.
No fallback ever substitutes a historical model for a formal model.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RECORDS = REPO / 'scripts/restructure'
# Prefix stored inside immutable JSON and checkpoint strings from the
# pre-restructure tree. relocate_string strips it; it is not a runtime data root.
OLD_REPO = '/home/plus7/Research/EEG_v1/EEG'

@lru_cache(maxsize=1)
def _moves():
    return sorted(json.loads((RECORDS/'moves.json').read_text()),
                  key=lambda row: len(row['old']), reverse=True)

def relocate_string(value: str) -> str:
    """Map a path string, preserving relative/absolute form when possible."""
    absolute = value.startswith(OLD_REPO+'/') or value.startswith(str(REPO)+'/')
    local = value
    for root in {OLD_REPO, str(REPO)}:
        if local.startswith(root+'/'):
            local = local[len(root)+1:]
            break
    prior = 'results/alljoined_leakfree_revision/matched_image_buffered_v2'
    if local == prior or local.startswith(prior+'/'):
        local = 'results/alljoined'+local[len(prior):]
    # Canonical destinations must never be interpreted as legacy checkpoints.
    if not any(local == row['new'] or local.startswith(row['new']+'/')
               for row in _moves()):
        for row in _moves():
            if row['old'] == 'checkpoints' and not local.startswith(('checkpoints/encoder/', 'checkpoints/predictor/')):
                continue
            if local == row['old'] or local.startswith(row['old']+'/'):
                local = row['new']+local[len(row['old']):]
                break
    if local == value and not absolute:
        return value
    return str(REPO/local) if absolute else local

def resolve_path(value: str | os.PathLike) -> Path:
    path = Path(relocate_string(str(value)))
    if not path.is_absolute():
        path = REPO/path
    return path.resolve()

def relocate_values(value):
    if isinstance(value, str):
        return relocate_string(value)
    if isinstance(value, list):
        return [relocate_values(item) for item in value]
    if isinstance(value, dict):
        # Hash/provenance dictionary keys are original identifiers, not editable paths.
        return {key: relocate_values(item) for key, item in value.items()}
    return value

def load_json(value, *args, **kwargs):
    return relocate_values(json.loads(value, *args, **kwargs))

def load_json_file(stream, *args, **kwargs):
    return relocate_values(json.load(stream, *args, **kwargs))

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def verify_source(path, expected):
    """Verify a frozen source or its explicitly recorded migration bridge.

    Both the original bytes and the exact post-migration bytes are checked;
    arbitrary later source edits still fail the experiment's freeze guard.
    """
    current = resolve_path(path)
    if current.is_file() and sha256(current) == expected:
        return
    bridge_file = RECORDS/'source_hash_bridge.json'
    if bridge_file.exists():
        bridge = json.loads(bridge_file.read_text())
        row = bridge.get(str(current.relative_to(REPO)))
        if row and row['before_sha256'] == expected:
            original = REPO/row['original_copy']
            if sha256(original) == expected and sha256(current) == row['after_sha256']:
                return
    raise RuntimeError(f'Frozen source does not match recorded migration: {path}')
