"""Portable path defaults for external EEG datasets.

Machine-specific roots belong in ``data/env.local`` (gitignored) or the
environment. Code defaults never hard-code ``/home/ubuntu/...``.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def eeg_data_root() -> str:
    return os.environ.get(
        'UBP_EEG_DATA_ROOT',
        str(Path.home() / 'datasets' / 'EEG'),
    )


def repo_root() -> Path:
    return Path(os.environ.get('UBP_REPO_ROOT', REPO_ROOT))
