.PHONY: setup test train backtest sweep paper-trade train-rl data-synthetic scrape-ft build-embeddings clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "Activate: source .venv/bin/activate"

test:
	pytest tests/ -v --tb=short

# Data
data-synthetic:
	python scripts/download_data.py --synthetic --days 90

scrape-ft:
	python scripts/scrape_ft.py --start 2024-01 --min-relevance 0.01

build-embeddings:
	python scripts/build_embeddings.py --data-dir data --cache-dir data/ft_cache/processed

# Training
train:
	python scripts/train_supervised.py model=dual_branch data=xauusd

train-hybrid:
	python scripts/train_supervised.py model=dual_branch data=xauusd data.labeling.method=hybrid

train-rl:
	python scripts/train_rl.py \
		--checkpoint models/dual_branch_best.pt \
		--steps 500000 \
		--max-hold 120 \
		--confidence-gate 0.3 \
		--episode-len 2000 \
		--eval-every 10000

train-rl-quick:
	python scripts/train_rl.py \
		--checkpoint models/dual_branch_best.pt \
		--steps 200000 \
		--confidence-gate 0.35 \
		--eval-every 5000

train-rl-long:
	python scripts/train_rl.py \
		--checkpoint models/dual_branch_best.pt \
		--steps 1000000 \
		--max-hold 120 \
		--confidence-gate 0.3 \
		--eval-every 20000

# Evaluation
backtest:
	python scripts/backtest.py model=dual_branch data=xauusd

backtest-filtered:
	python scripts/backtest.py model=dual_branch data=xauusd ++min_confidence=0.5

backtest-strict:
	python scripts/backtest.py model=dual_branch data=xauusd ++min_confidence=0.7

backtest-hitl:
	python scripts/backtest.py model=dual_branch data=xauusd ++risk.human_exit_approval=true ++min_confidence=0.5

# Sweep
sweep:
	wandb sweep configs/sweep.yaml

# Paper trading
paper-trade:
	python scripts/paper_trade.py --synthetic

paper-trade-live:
	python scripts/paper_trade.py --config configs/deployment/production.yaml

# Clean
clean:
	rm -rf __pycache__ .pytest_cache outputs/ multirun/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
