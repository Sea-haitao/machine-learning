# DSAA2011 Machine Learning Project

Predict student academic outcomes (Dropout / Graduate / Enrolled) using machine learning.

**Deadline: May 10, 2026**

## Setup

```bash
git clone git@github.com:Sea-haitao/machine-learning.git
cd machine-learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Phases

| Phase | Task | Deadline |
|-------|------|----------|
| 1 | Data preprocessing | Apr 9 |
| 2 | Data visualization | Apr 9 |
| 3 | Clustering analysis | TBD |
| 4 | Prediction: training & testing | TBD |
| 5 | Evaluation & model selection | TBD |
| 6 | Open-ended exploration | TBD |
| 7 | Report, PPT & final notebook | May 10 |

Each phase is owned by two people. Present results at the next meeting, then discuss and assign the next task.

## Git Workflow

### 1. Before you start working

Always sync with main first:

```bash
git checkout main
git pull
```

### 2. Create a feature branch

Name it after the task you're working on:

```bash
git checkout -b feat/preprocessing
```

Branch naming examples:
- `feat/preprocessing` — data cleaning and feature engineering
- `feat/visualization` — EDA plots
- `feat/clustering` — clustering analysis
- `feat/prediction` — model training
- `feat/evaluation` — model comparison
- `feat/exploration` — open-ended analysis

### 3. Work and commit often

```bash
git add notebooks/01_preprocessing.ipynb src/preprocess.py
git commit -m "Add missing value handling and feature encoding"
```

Write clear commit messages describing **what you did**, not "update notebook".

### 4. Push and open a Pull Request

```bash
git push -u origin feat/preprocessing
```

Then open a PR on GitHub. Add a short description of what you did and tag a teammate for review.

### 5. Review and merge

- Teammate reviews the PR (a quick look is fine — check that it runs and makes sense)
- Merge into `main` on GitHub
- Delete the feature branch after merging

## Notebook Rules

Jupyter notebooks cause messy merge conflicts. Follow these rules:

1. **One person edits a notebook at a time.** Never have two people working on the same `.ipynb` on different branches.
2. **Number notebooks by phase:** `01_preprocessing.ipynb`, `02_visualization.ipynb`, etc.
3. **Extract reusable code into `src/`.** Notebooks should call functions from `src/`, not duplicate logic.

## Final Notebook Merge

Before submission, all phase notebooks are merged into one `final.ipynb`:

1. **Freeze all phase notebooks.** Make sure every notebook in `notebooks/` runs top-to-bottom without errors (`Kernel → Restart & Run All`).
2. **Create a merge branch:**
   ```bash
   git checkout main && git pull
   git checkout -b feat/final-notebook
   ```
3. **Run the merge script:**
   ```bash
   python scripts/merge_notebooks.py
   ```
   This concatenates all numbered notebooks (`01_*.ipynb`, `02_*.ipynb`, ...) in order into `final.ipynb` at the project root.
4. **Review and clean up `final.ipynb`:**
   - Open it in Jupyter, run all cells to verify
   - Add transition text/markdown cells between sections if needed
   - Remove any duplicate imports (keep one import cell at the top)
5. **Commit, push, and open a PR** for team review before submission.

> **Important:** Do not edit `final.ipynb` directly during the project. Always work in the individual phase notebooks. Only generate `final.ipynb` at the end.

## Project Structure

```
data/           → raw dataset (do not modify data.csv directly)
notebooks/      → one Jupyter notebook per phase
src/            → shared Python functions (preprocessing, model utils, etc.)
outputs/        → saved figures, model files, results
```

## Meeting Flow

1. Present task progress (~15 min)
2. Q&A and discussion (~10 min)
3. Assign next task, confirm owners and deadline (~5 min)

## Dataset

- File: `data/data.csv` (semicolon `;` separated)
- Load with: `pd.read_csv('data/data.csv', sep=';')`
- Target column: `Target` (Dropout / Graduate / Enrolled)
- 36 features: demographics, academic performance, macroeconomic indicators
