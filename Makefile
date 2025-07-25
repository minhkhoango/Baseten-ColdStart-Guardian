# Makefile for Baseten Cold Start Predictor & Warmer

.PHONY: setup run format

setup:
	pip install poetry
	poetry install
	poetry add --group dev pandas-stubs black

run:
	poetry run python cold_start_warmer.py

format:
	poetry run black cold_start_warmer.py 