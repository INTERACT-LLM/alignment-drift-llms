### INDIVIDUAL COMMANDS ###
add-uv:
	@echo "[INFO:] Installing UV ..."	
	# for mac / linux
	curl -LsSf https://astral.sh/uv/install.sh | sh

install:
	@echo "[INFO:] Installing project ..."
	uv sync

format: 
	@echo "[INFO:] Formatting code with ruff ..."
	uv run ruff format . 						           
	uv run ruff check --select I --fix

create_dataset:
	@echo "[INFO:] Creating dataset ..."
	uv run src/create_dataset.py
	@echo "[INFO:] Creating dataset done."

extract_%: # usage: make extract_all, make extract_textdescriptives, make extract_surprisal or make extract_textstats
	@echo "[INFO:] Extracting $* ..."
	uv run src/extract_metrics.py --metrics_pipeline $*

plot:
	@echo "[INFO:] Plotting ..."
	uv run src/plots.py


### PROJECT PIPELINE ###
setup-project:
	make add-uv
	make install

run-code: # note stats.rmd is not run
	make extract_all
	make plot

run-project:
	make setup-project
	make run-code