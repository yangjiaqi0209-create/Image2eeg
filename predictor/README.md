# predictor — image → EEG

Train the image-conditioned EEG predictor (uses a frozen encoder from `encoder/`).

```bash
python -m predictor.train --sub 1 --gpu 0
# or
bash scripts/train_final_two_stage.sh
bash scripts/train_alljoined_final_two_stage.sh
```

Weights → `checkpoints/predictor/`. Hyperparameters are CLI / shell scripts (no yaml under `configs/eeg/`).
