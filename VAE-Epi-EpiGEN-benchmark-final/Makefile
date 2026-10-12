PYTHON ?= python

.PHONY: test demo smoke-test

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) scripts/generate_demo_data.py --outdir data/example

smoke-test: demo
	$(PYTHON) scripts/01_calibrate_epigen_alpha.py \
		--genotype data/example/genotype.csv \
		--pairs data/example/causal_pairs.csv \
		--target 0.10 \
		--noise_sd 1.0 \
		--out data/example/calibration
