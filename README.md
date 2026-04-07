# DSAA2011 Machine Learning Project

Predict student academic outcomes (Dropout / Graduate / Enrolled) using machine learning.

**Deadline: May 10, 2026 (11:59pm Beijing Time)**
**Presentation: May 11, 2026 (last lecture session, 5 min + 3 min Q&A)**

## Setup

```bash
git clone git@github.com:Sea-haitao/machine-learning.git
cd machine-learning
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Phases

| Phase | Task | Key Requirements | Deadline |
|-------|------|------------------|----------|
| 1 | Data Preprocessing | Handle missing values, encode non-numeric values, standardize features | Apr 9 |
| 2 | Data Visualization | **t-SNE** 2D/3D scatter plot, color by class label, analyze patterns | Apr 9 |
| 3 | Clustering Analysis | **At least 2** algorithms (e.g. K-means, hierarchical), multiple metrics, compare & visualize | TBD |
| 4 | Prediction | **At least 2** models (e.g. decision tree, logistic regression), 70/30 split, evaluate on **train/test/entire** set, confusion matrices | TBD |
| 5 | Evaluation | Accuracy/precision/recall/F1, **ROC + AUC** for each model, improve via validation, 100-200 word discussion | TBD |
| 6 | Open-ended Exploration | e.g. model improvement, compare 3+ models with cross-validation, feature engineering, hyperparameter tuning | TBD |
| 7 | Report, PPT & final notebook | See Submission below | May 10 |

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
- `feat/visualization` — t-SNE plots
- `feat/clustering` — clustering analysis
- `feat/prediction` — model training
- `feat/evaluation` — model comparison, ROC/AUC
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
4. **Use markdown cells** to describe your approach, explain each step, and provide context for visualizations.
5. **Label all plots** with titles, axis labels, and legends.

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
5. **Verify reproducibility:** `Kernel → Restart & Run All` must complete without errors.
6. **Commit, push, and open a PR** for team review before submission.

> **Important:** Do not edit `final.ipynb` directly during the project. Always work in the individual phase notebooks. Only generate `final.ipynb` at the end.

## Submission Packaging

The final submission is a zip file uploaded to Canvas. Run the packaging script or manually assemble:

```
groupID_datasetB.zip
├── presentation_groupID_datasetB.pdf   ← slides (PDF)
├── report_groupID_datasetB.pdf         ← report (PDF, LaTeX required, 5-10 pages)
├── project_groupID_datasetB.ipynb      ← final merged notebook (must run start-to-finish)
├── requirements_groupID_datasetB.txt   ← pip freeze output
└── data/                               ← (optional) dataset
```

Replace `groupID` with your assigned group ID. Report must follow the required LaTeX style file and this structure:
1. Introduction
2. Mandatory Tasks (t-SNE, clustering, model training, confusion matrices)
3. Open-ended Exploration
4. Conclusion
5. References
6. Credit (contribution of each member + GenAI tool usage)

## GenAI Usage Policy

- GenAI tools (LLMs) may be used for **<30%** of the work — support tasks like debugging, generating visualizations, drafting text.
- **All core ML implementations** (t-SNE, clustering, model training, analysis) must be your own work.
- **You must disclose** GenAI usage in the report's References section, specifying the tool and how it was applied.

## Project Structure

```
data/           → raw dataset (do not modify data.csv directly)
notebooks/      → one Jupyter notebook per phase
src/            → shared Python functions (preprocessing, model utils, etc.)
scripts/        → utility scripts (e.g. merge_notebooks.py)
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
