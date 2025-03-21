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

extract_surprisal:
	@echo "[INFO:] Extracting surprisal values ..."
	uv run src/extract_metrics.py --metrics_pipeline surprisal

extract_all:
	@echo "[INFO:] Extracting all metrics ..."
	uv run src/extract_metrics.py --metrics_pipeline all

extract_textdescriptives:
	@echo "[INFO:] Extracting text descriptives ..."
	uv run src/extract_metrics.py --metrics_pipeline textdescriptives

extract_textstats:
	@echo "[INFO:] Extracting text stats ..."
	uv run src/extract_metrics.py --metrics_pipeline textstats