.PHONY: setup data synthetic smoke run run-quality test

setup:
	pip install -r requirements.txt

data:
	bash scripts/download_mind.sh data/MINDsmall

synthetic:
	python scripts/make_synthetic.py --out data/synthetic

smoke: synthetic
	python scripts/run_pipeline.py --data-dir data/synthetic --quick --out results_smoke

run:
	python scripts/run_pipeline.py --data-dir data/MINDsmall --out results

# Needs data/mind_quality.tsv from the content-quality-classifier repo
run-quality:
	python scripts/run_pipeline.py --data-dir data/MINDsmall --out results \
		--quality-scores data/mind_quality.tsv

test:
	python -m pytest -q
