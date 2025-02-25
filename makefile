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
	ruff check --select I --fix