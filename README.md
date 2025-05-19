![alignment-drift-header](https://github.com/user-attachments/assets/724bb5fd-613f-4dac-8c4e-a980504dd388)


# 🚀 Overview  
This repository contains the dataset and analysis from [Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351):

| Item                    | Location                                      | Documentation                   |
|-------------------------|--------------------------------------------------------|--------------------------------|
| 📦 Text Dataset (`v3.0`)       | [`data/v3.0_dataset.csv`](data/v3.0_dataset.csv) | [`data/README.md`](data/README.md)         |
| 📦 Metrics Dataset (`v3.0`)       | [`metrics/*.csv`](metrics) | [`metrics/README.md`](data/README.md)         |
| 🧪 Analysis               | [`src/`](src/)                                | [`src/README.md`](src/README.md)           |
| 📊 Plots & Results        | [`plots/`](plots/) & [`results/`](results/) |        |

<br>

Teacher-student dialogue simulations were performed in a separate repository:

| Item                    | Location                                                  | Documentation                         |
|-------------------------|-----------------------------------------------------------|------------------------------------|
| 🛠️ Generation of Dialogues | [`Interact-LLM repo (src/scripts/alignment-drift)`](https://github.com/INTERACT-LLM/Interact-LLM) | [`README.md`](https://github.com/INTERACT-LLM/Interact-LLM#readme) |


<span style="display: block; margin-top: 30px;"></span>
> Note: The prefix `v3.0` for the data refers to the prompt version used to simulate the dialogues. See the prompts in the [Interact-LLM repo](https://github.com/INTERACT-LLM/Interact-LLM/blob/main/configs/prompts/v3.0.toml).

# 🛠️ Technical Requirements
The code was run on `Python 3.12.3` on both a macOS (`15.3.1`) and Ubuntu system (`24.04`). The project also requires:
| Tool     | Installation                                                                 |
|----------|--------------------------------------------------------------------------------------|
| [make](https://www.gnu.org/software/make/manual/make.html) | Installed via [Homebrew](https://formulae.brew.sh/formula/make)                  |
| [uv](https://docs.astral.sh/uv/)                         | Installed through this project's `makefile` (see [Usage](#usage))                 |
| [R 4.4.3](https://cran.r-project.org/bin/macosx/big-sur-arm64/base/) + R Markdown           | Installed separately via [CRAN](https://cran.r-project.org/bin/macosx/big-sur-arm64/base) for R and [Posit's RStudio](https://docs.posit.co/previous-versions/rstudio.html#section-1) for running R-Markdown (or an IDE of your liking).                                |


<a name="usage"></a>

# ⚙️ Usage
You can run the code using the [`makefile`](makefile) by entering the following command in the terminal:
```bash
make run-project
```

This command installs [`uv`](https://docs.astral.sh/uv/) on macOS/Linux, sets up a virtual environment with the required Python dependencies, and finally runs the code.

If you prefer to run your own installation of [`uv`](https://docs.astral.sh/uv/) (or already have it installed), you can run only the code directly:
```bash
make run-code
```

<span style="display: block; margin-top: 20px;"></span>

> Note: This does not execute `stats.rmd.`. It must be run seperately (requires R and R Markdown, see [Technical Requirements](#️-technical-requirements)).


# 📝 Citation 
If you use our work, please remember to cite us:

```
@article{almasi2025alignmentdriftcefrpromptedllms,
  title={Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring}, 
  author={Mina Almasi and Ross Deans Kristensen-McLachlan},
  journal={arXiv preprint arXiv:2505.08351},
  year={2025},
  url={https://arxiv.org/abs/2505.08351},
  note={cs.CL}
}
```

<span style="display: block; margin-top: 20px;"></span>

> Note: Currently, this work exists only as a preprint. The final version is forthcoming.

# ✨ Acknowledgements
This work was made possible thanks to the following open-source resources:

- [textstat](https://textstat.org/) for Spanish Readability Metrics
- [textdescriptives](https://hlasse.github.io/TextDescriptives/#citation) for Text Length & Mean Dependency Distance
- [minicons](https://github.com/kanishkamisra/minicons?tab=readme-ov-file#citation) & [EuroBERT](https://huggingface.co/EuroBERT/EuroBERT-210m#citation) for LLM-based Message Surprisal

See also [`metrics/README.md`](metrics/README.md).