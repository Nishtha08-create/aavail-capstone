# AAVAIL Revenue Prediction - Enterprise AI Workflow

This repository contains the end-to-end containerized solution for predicting daily revenue across various countries for AAVAIL.

## Project Architecture
- `ingest.py`: Automated data loading and country-level time-series aggregation.
- `model.py`: Time-series lag feature extraction, linear baseline evaluation, and Random Forest production training.
- `logger.py`: Production and isolated test log auditing.
- `app.py`: Flask REST API serving `/predict` endpoints for specific countries or global queries.
- `run_tests.py`: Master test suite executing API, model, data ingestion, and logging tests.
- `notebooks/aavail_capstone_notebook.ipynb`: EDA and baseline model comparison plots.

## Running Tests
```bash
python run_tests.py