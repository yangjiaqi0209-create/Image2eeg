"""Redraw paper figures final_fig1–4 and S_fig1–4 from cached raw arrays.

Requires ``analysis/eeg_gen_eval/raw`` (THINGS) and ``raw_alljoined`` (S_fig4).
Each figure runs in a subprocess so THINGS/Alljoined config switches stay isolated.

Usage:
  PYTHONPATH=. python -m analysis.eeg_gen_eval.plots.redraw_all
  PYTHONPATH=. python -m analysis.eeg_gen_eval.plots.redraw_all --only final_fig1,S_fig4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Tuple

FIGURES: Dict[str, Tuple[str, str]] = {
    'final_fig1': (
        'analysis.eeg_gen_eval.plots.plot_final_fig1',
        'plot_final_fig1_eeg_response_prediction',
    ),
    'final_fig2': (
        'analysis.eeg_gen_eval.plots.plot_final_fig2',
        'plot_final_fig2_frequency_spectral',
    ),
    'final_fig3': (
        'analysis.eeg_gen_eval.plots.plot_final_fig3',
        'plot_final_fig3_representational_alignment',
    ),
    'final_fig4': (
        'analysis.eeg_gen_eval.plots.plot_final_fig4',
        'plot_final_fig4_ablations',
    ),
    'S_fig1': (
        'analysis.eeg_gen_eval.plots.plot_s_fig1',
        'plot_s_fig1_prediction_quality_supp',
    ),
    'S_fig2': (
        'analysis.eeg_gen_eval.plots.plot_s_fig2',
        'plot_s_fig2_single_image_waveforms',
    ),
    'S_fig3': (
        'analysis.eeg_gen_eval.plots.plot_s_fig3',
        'plot_s_fig3_representational_alignment_supp',
    ),
    'S_fig4': (
        'analysis.eeg_gen_eval.plots.plot_s_fig4_alljoined_montage',
        'plot_s_fig4_alljoined_montage',
    ),
}


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # plots/ -> eeg_gen_eval/ -> analysis/ -> repo
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _run_one(key: str) -> None:
    mod_name, fn_name = FIGURES[key]
    code = (
        f'from {mod_name} import {fn_name}; '
        f'out = {fn_name}(); '
        f'print(out or {key!r}, "done")'
    )
    print(f'=== {key} ===', flush=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = _repo_root()
    subprocess.run(
        [sys.executable, '-c', code],
        check=True,
        cwd=_repo_root(),
        env=env,
    )


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Redraw final/S paper figures from cache')
    p.add_argument(
        '--only',
        type=str,
        default='',
        help='Comma-separated subset, e.g. final_fig1,S_fig4',
    )
    args = p.parse_args(argv)
    keys = list(FIGURES)
    if args.only.strip():
        keys = [k.strip() for k in args.only.split(',') if k.strip()]
        unknown = [k for k in keys if k not in FIGURES]
        if unknown:
            raise SystemExit(f'Unknown figure keys: {unknown}; choose from {list(FIGURES)}')

    for key in keys:
        _run_one(key)
    print('All requested figures redrawn.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
