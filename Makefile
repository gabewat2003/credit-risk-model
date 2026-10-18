.PHONY: features train evaluate cutoff all

features:
	uv run python src/build_features.py

train:
	uv run python src/train.py

evaluate:
	uv run python src/evaluate.py

cutoff:
	uv run python src/cutoff.py

all: features train evaluate cutoff
