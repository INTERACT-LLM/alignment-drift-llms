# 📊 Metrics Overview

This folder contains the metrics extracted from the consolidated text dataset [`data/v3.0_dataset.csv`](data/v3.0_dataset.csv).

| File                        | Description                                                                                      |
|-----------------------------|------------------------------------------------------------------------------------------------|
| `v3.0_surprisal.csv`        | Message surprisal computed by [EuroBERT (210m)](https://huggingface.co/EuroBERT/EuroBERT-210m#citation) using [minicons](https://github.com/kanishkamisra/minicons?tab=readme-ov-file#citation)           |
| `v3.0_text_stats.csv`       | Spanish readability metrics using [textstat](https://textstat.org/)                             |
| `v3.0_textdescriptives.csv` | Mean Dependency Distance, Text Length, and other features extracted with [textdescriptives](https://hlasse.github.io/TextDescriptives/#citation) |

For details on how metrics were computed, refer to our paper: [Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351)