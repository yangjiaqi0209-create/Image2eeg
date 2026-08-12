# Manuscript Results figures

```text
plots/      # final_fig1–4, S_fig1–4, redraw_all
helpers/    # shared drawing utilities (style, panels, embedding, …)
compute/    # metrics, evaluate, RSA / ablation aggregation
config.py   # paths + dataset switch (THINGS / Alljoined)
raw/        # THINGS caches   |  raw_alljoined/  Alljoined caches
figures/    # published PNG/SVG outputs
```

```bash
# Redraw all paper figures from caches
PYTHONPATH=. python -m analysis.eeg_gen_eval.plots.redraw_all
# or shim:
PYTHONPATH=. python -m analysis.eeg_gen_eval.redraw_all --only final_fig1,S_fig4
```
