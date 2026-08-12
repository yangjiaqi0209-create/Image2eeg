"""Shim: prefer ``python -m analysis.eeg_gen_eval.plots.redraw_all``."""

from analysis.eeg_gen_eval.plots.redraw_all import FIGURES, main

if __name__ == '__main__':
    raise SystemExit(main())
