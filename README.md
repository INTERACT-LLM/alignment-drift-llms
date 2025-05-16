![alignment-drift-header](https://github.com/user-attachments/assets/724bb5fd-613f-4dac-8c4e-a980504dd388)


# 🚀 Overview  
This repository contains the dataset and analysis from [Almasi & Kristensen-McLachlan (2025)](https://arxiv.org/abs/2505.08351):

| Item                    | Path                                      | Documentation                   |
|-------------------------|--------------------------------------------------------|--------------------------------|
| 📦 Dataset (`v3.0`)       | [`data/v3.0_dataset.csv`](data/v3.0_dataset.csv) | [`data/README.md`](data/README.md)         |
| 🧪 Analysis               | [`src/`](src/)                                | [`src/README.md`](src/README.md)           |
| 📊 Plots & Results        | [`plots/`](plots/) & [`results/`](results/) |        |

Teacher-student dialogue simulations were performed in a separate repository:

| Item                    | Path                                                  | Documentation                         |
|-------------------------|-----------------------------------------------------------|------------------------------------|
| 🛠️ Data Generation | [interact_llm/scripts/alignment_drift](https://github.com/INTERACT-LLM/Interact-LLM) | [alignment_drift/README.md](https://github.com/INTERACT-LLM/Interact-LLM#readme) |

---

# 🛠️ Technical Requirements
The code was run on Python 3.12.3 on both a macOS and Ubuntu system.

This project requires the installation of 
- [make](https://www.gnu.org/software/make/manual/make.html)&nbsp; (installed via [Homebrew](https://formulae.brew.sh/formula/make))
- [uv](https://docs.astral.sh/uv/)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (installed through this project's make file, see [Usage](#usage))

<a name="usage"></a>

# ⚙️ Usage
You can run the code using the [makefile](makefile) (requires `make`, which can be installed via [Homebrew](https://formulae.brew.sh/formula/make)):
```bash
make run-project
```

This command installs [uv](https://docs.astral.sh/uv/) on macOS/Linux and sets up a virtual environment with the required Python dependencies.

If you prefer to run your own installation of uv (or already have it installed), you can run only the code directly:
```bash
make run-code
```
Note: This does not execute stats.rmd. To run that file, you will need R and R Markdown installed. It must be run separately.


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

NOTE: Currently, this work exists only as a preprint. The final version is forthcoming.
