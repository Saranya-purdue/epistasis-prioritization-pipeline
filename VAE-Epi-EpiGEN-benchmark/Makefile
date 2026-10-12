PYTHON ?= python

.PHONY: test demo clean

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) scripts/generate_demo_data.py --outdir demo_data
	$(PYTHON) scripts/01_calibrate_epigen_alpha.py \
		--genotype demo_data/genotype.csv \
		--pairs demo_data/causal_pairs.csv \
		--target 0.10 \
		--noise_sd 1.0 \
		--out demo_data/calibration

clean:
	rm -rf demo_data .pytest_cache
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
