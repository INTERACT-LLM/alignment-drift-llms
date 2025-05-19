# 🧪 + 📊 Code Overview

Each file in the `src` directory contributes to the dataset analysis pipeline. Here's a breakdown:

| File/Folder                   | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `create_dataset.py`          | Creates the consolidated dataset from individual JSON files for each model (see [data/README.md](../data/README.md))       |
| `extract_metrics.py`         | Extracts evaluation metrics (see [metrics/README.md](../metrics/README.md))                    |
| `plots.py`                   | Central plotting script for visualising results                            |
| `stats.Rmd`                  | R Markdown file for running linear mixed effects models               |
| `utils/metrics_process.py`   | Functions for reading metrics files, filtering and aggregating                        |
| `utils/plot_functions.py`    | Reusable plotting functions (e.g. styling, formatting) used in `plots.py`                     |
| `utils/text_process.py`      | Utilities for removing emojis, splitting paragraphs into sents for surprisal metric    |


🛠️ See also the [technical requirements](/README.md#️-technical-requirements) for running the code.