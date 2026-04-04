# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DSAA2011 course ML project (Spring 2026). The goal is to analyze a student academic performance dataset (`data.csv`, ~4424 rows, semicolon-delimited) predicting student outcomes (Dropout/Graduate/Enrolled). The project covers data preprocessing, visualization, clustering, prediction modeling, evaluation, and open-ended exploration.

## Environment

- Python 3.10 virtualenv at `.venv/`
- Activate: `source .venv/bin/activate`
- Key libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, jupyter/jupyterlab

## Commands

```bash
# Start Jupyter
source .venv/bin/activate && jupyter lab

# Run a Python script
source .venv/bin/activate && python script.py

# Install new packages
source .venv/bin/activate && pip install <package>
```

## Dataset

- File: `data/data.csv` (semicolon `;` separator, not comma)
- Load with: `pd.read_csv('data/data.csv', sep=';')`
- Target column: `Target` (values: Dropout, Graduate, Enrolled)
- 36 feature columns covering demographics, academic performance (1st/2nd semester curricular units), and macroeconomic indicators
