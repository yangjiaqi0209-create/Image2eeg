# `data/` — datasets + files

Common layout: **registry/profiles** (small, tracked) sit next to **THINGS files** (large, gitignored).

```text
data/
├── registry.yaml              # dataset index
├── profiles/                  # things_eeg.yaml, alljoined_eeg.yaml
├── env.example                # UBP_EEG_DATA_ROOT, …
└── things-eeg/                # in-repo THINGS EEG / images / CLIP features
    ├── Preprocessed_data_250Hz_whiten/
    ├── Image_set_Resize/
    └── Image_feature/
```

Alljoined `.pt` live outside the repo: `$UBP_EEG_DATA_ROOT/alljoined-1.6M/`（说明见根 `README.md` 的 Alljoined 小节）。

```bash
python -m encoder.registry list
python -m encoder.registry show things_eeg
python -m encoder.registry show alljoined_eeg
```
