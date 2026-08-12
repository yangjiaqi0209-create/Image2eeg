# encoder — EEG brain encoder (UBP)

Train the CLIP-aligned EEG encoder.

```bash
python -m encoder.train --config configs/eeg/fixed_fovea.yaml ...
# or
bash scripts/train_things_encoder.sh
bash scripts/train_alljoined.sh
```

Weights → `checkpoints/encoder/`. Dataset profiles: `python -m encoder.registry list`.
