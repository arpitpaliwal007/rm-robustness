.PHONY: data baselines test train probe report all clean

DATA_DIR ?= data
OUT      ?= runs/base
BACKBONE ?= distilroberta-base

data:
	python scripts/fetch_data.py --data-dir $(DATA_DIR)

baselines:
	python scripts/day0_baselines.py --data-dir $(DATA_DIR)

test:
	python -m pytest tests -q

# a two-minute end-to-end run with the offline backbone: proves the pipeline, not the science
smoke:
	python -m rmrobust.cli all --backbone tiny --max-length 192 --limit-train 1500 \
	  --limit-eval 200 --max-steps 40 --eval-every 20 --lr 3e-4 --batch-size 8 \
	  --grad-accum 1 --no-amp --n-boot 300 --max-position-pairs 80 \
	  --max-perturbation-pairs 80 --quiet --out runs/smoke

train:
	python -m rmrobust.cli train --backbone $(BACKBONE) --out $(OUT)

probe:
	python -m rmrobust.cli probe --checkpoint $(OUT)/best --out $(OUT) --quiet

all: train probe

clean:
	rm -rf runs/*/figures runs/*/results.json runs/*/report.md
