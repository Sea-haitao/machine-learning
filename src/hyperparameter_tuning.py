"""
Hyperparameter Tuning Module for Student Outcome Prediction

This module performs comprehensive hyperparameter optimization for multiple models:
- Decision Tree, Logistic Regression, Random Forest, Gradient Boosting, SVC, MLP
- Includes class_weight='balanced' to address class imbalance
- Uses Pipeline with StandardScaler for fair solver comparison
- Uses StratifiedKFold with shuffle for reproducible CV

Best used when imported from the notebook via:
    from src.hyperparameter_tuning import run_tuning_pipeline
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# Label mapping for readability
LABEL_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

# Random seed for reproducibility
RANDOM_STATE = 42


def evaluate_model(model_name, search, x_test, y_test, label_names=None):
    """Evaluate the best model from GridSearchCV on test data."""
    y_pred = search.predict(x_test)
    target_names = (
        [label_names[i] for i in sorted(label_names.keys())] if label_names else None
    )
    return {
        "model": model_name,
        "best_params": search.best_params_,
        "cv_best_f1_macro": search.best_score_,
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_names
        ),
    }


def run_grid_search(estimator, param_grid, x_train, y_train, cv, scoring="f1_macro"):
    """Run GridSearchCV with consistent settings."""
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search


def run_tuning_pipeline():
    """Main pipeline for hyperparameter tuning."""
    # Setup paths
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "data_preprocessed.csv"
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)
    x = df.drop(columns=["Target", "Target_encoded"])
    y = df["Target_encoded"]

    # Stratified train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    # Consistent CV strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # =====================================================================
    # Model 1: Decision Tree
    # =====================================================================
    dt_params = {
        "max_depth": [3, 5, 8, 12, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "criterion": ["gini", "entropy", "log_loss"],
        "class_weight": [None, "balanced"],
    }
    dt_search = run_grid_search(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=dt_params,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
    )

    # =====================================================================
    # Model 2: Logistic Regression (with StandardScaler Pipeline)
    # =====================================================================
    lr_pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(random_state=RANDOM_STATE, max_iter=2000)),
        ]
    )
    lr_params = {
        "lr__C": [0.01, 0.1, 1, 3, 10],
        "lr__solver": ["lbfgs", "newton-cg", "saga"],
        "lr__class_weight": [None, "balanced"],
    }
    lr_search = run_grid_search(
        estimator=lr_pipe,
        param_grid=lr_params,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
    )

    # =====================================================================
    # Model 3: Random Forest
    # =====================================================================
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": [None, "balanced", "balanced_subsample"],
    }
    rf_search = run_grid_search(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=rf_params,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
    )

    # =====================================================================
    # Model 4: Gradient Boosting
    # =====================================================================
    gb_params = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.8, 1.0],
    }
    gb_search = run_grid_search(
        estimator=GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid=gb_params,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
    )

    # =====================================================================
    # Model 5: SVC (with StandardScaler Pipeline)
    # =====================================================================
    svc_pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("svc", SVC(random_state=RANDOM_STATE)),
        ]
    )
    svc_params = {
        "svc__C": [0.1, 1, 10],
        "svc__kernel": ["rbf", "linear"],
        "svc__gamma": ["scale", "auto"],
        "svc__class_weight": [None, "balanced"],
    }
    svc_search = run_grid_search(
        estimator=svc_pipe,
        param_grid=svc_params,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
    )

    # =====================================================================
    # Model 6: MLP (with StandardScaler Pipeline)
    # =====================================================================
    mlp_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    random_state=RANDOM_STATE, max_iter=500, early_stopping=True
                ),
            ),
        ]
    )
    mlp_params = {
        "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__alpha": [0.0001, 0.001, 0.01],
        "mlp__learning_rate": ["constant", "adaptive"],
    }
    mlp_search = run_grid_search(
        estimator=mlp_pipe,
        param_grid=mlp_params,
        x_train=x_train,
        y_train=y_train,
        cv=cv,
    )

    # =====================================================================
    # Bonus: Random Forest with SMOTE
    # =====================================================================
    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    rf_smote_search = run_grid_search(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=rf_params,
        x_train=x_train_smote,
        y_train=y_train_smote,
        cv=cv,
    )

    # =====================================================================
    # Evaluate all models
    # =====================================================================
    results = [
        evaluate_model("DecisionTree", dt_search, x_test, y_test, LABEL_NAMES),
        evaluate_model("LogisticRegression", lr_search, x_test, y_test, LABEL_NAMES),
        evaluate_model("RandomForest", rf_search, x_test, y_test, LABEL_NAMES),
        evaluate_model("GradientBoosting", gb_search, x_test, y_test, LABEL_NAMES),
        evaluate_model("SVC", svc_search, x_test, y_test, LABEL_NAMES),
        evaluate_model("MLP", mlp_search, x_test, y_test, LABEL_NAMES),
        evaluate_model("RandomForest+SMOTE", rf_smote_search, x_test, y_test, LABEL_NAMES),
    ]

    # Sort by test F1-macro
    results.sort(key=lambda x: x["test_f1_macro"], reverse=True)

    # Create results DataFrame
    results_df = pd.DataFrame(
        [
            {
                "model": r["model"],
                "cv_best_f1_macro": r["cv_best_f1_macro"],
                "test_accuracy": r["test_accuracy"],
                "test_f1_macro": r["test_f1_macro"],
                "best_params": json.dumps(r["best_params"], ensure_ascii=False),
            }
            for r in results
        ]
    )

    # Save results
    csv_path = outputs_dir / "hyperparameter_tuning_results.csv"
    txt_path = outputs_dir / "hyperparameter_tuning_report.txt"
    json_path = outputs_dir / "hyperparameter_best_params.json"

    results_df.to_csv(csv_path, index=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Model: {r['model']}\n")
            f.write(f"Best Params: {r['best_params']}\n")
            f.write(f"Best CV F1-macro: {r['cv_best_f1_macro']:.4f}\n")
            f.write(f"Test Accuracy: {r['test_accuracy']:.4f}\n")
            f.write(f"Test F1-macro: {r['test_f1_macro']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(r["classification_report"])
            f.write("\n" + "=" * 80 + "\n\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({r["model"]: r["best_params"] for r in results}, f, indent=2)

    print(f"\nResults saved to: {csv_path}")
    print(f"Detailed report saved to: {txt_path}")
    print(f"Best params saved to: {json_path}")

    return results, results_df


if __name__ == "__main__":
    results, results_df = run_tuning_pipeline()
    print("\n" + "=" * 60)
    print("Final Results Summary:")
    print("=" * 60)
    print(results_df.to_string(index=False))
