PYTHON = python -m

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -e .

data:
	$(PYTHON) src.data.preprocessing

# ── Individual model training ─────────────────────────────────────────────────
 
train-rnn:
	$(PYTHON) scripts.train \
		--config   configs/rnn_large.yaml \
		--dataset $(DATASET)
 
train-gru:
	$(PYTHON) scripts.train \
		--config   configs/gru_large.yaml \
		--dataset $(DATASET)
 
train-lstm:
	$(PYTHON) scripts.train \
		--config   configs/lstm_large.yaml \
		--dataset $(DATASET)
 
train-transformer:
	$(PYTHON) scripts.train \
		--config   configs/transformer_large.yaml \
		--dataset $(DATASET)
 
# ── Train all models on a given dataset ───────────────────────────────────────
# Usage:
#   make train-all-on                          # uses DATASET=shakespeare
#   make train-all-on DATASET=my_corpus
 
train-all-on: train-rnn train-gru train-lstm train-transformer
 
# ── Train all on default dataset ──────────────────────────────────────────────
 
train-all: train-all-on

# ── Quick smoke test (CPU, few epochs) ───────────────────────────────────────

test-rnn:
	$(PYTHON) scripts.train \
		--config   configs/rnn_large.yaml \
		--epochs   2 \
		--device   cpu

test-transformer:
	$(PYTHON) scripts.train \
		--config   configs/transformer_large.yaml \
		--epochs   2 \
		--device   cpu

test-all: test-rnn test-transformer

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean-experiments:
	rm -rf experiments/

clean-data:
	rm -rf data/processed/

clean: clean-experiments clean-data

.PHONY: install data train-rnn train-gru train-lstm train-transformer \
        train-all test-rnn test-transformer test-all \
        clean clean-experiments clean-data
