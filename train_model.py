"""Reproduce the EM-VRF signed-ΔV model from Supplementary Table 5.

All 45 records in data/Supplementary Table 5.xlsx are used as provided. R_avg and Φ_TM are
taken from the table; ΔX_Na is recalculated from the formula and Q_ch without rounding.
Model: StandardScaler + RBF-SVR (C = 3, ε = 0.001, γ = "scale").

The script refits the model, repeats the 20 × 5 cross-validation (seed 20260912) and checks
both against the files used by the application (data/EM-VRF-ΔV.pkl, data/model_meta.json).
--full-validation also repeats the nested 10 × 5 cross-validation and the seed-sensitivity
runs. Results go to a new folder (default results/retrained); data/ is never modified.

usage: python train_model.py [--output DIR] [--full-validation]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import operator
import pickle
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET = DATA_DIR / "Supplementary Table 5.xlsx"
ELEMENTS = DATA_DIR / "element_database.csv"
MODEL = DATA_DIR / "EM-VRF-ΔV.pkl"
MODEL_META = DATA_DIR / "model_meta.json"

FARADAY = 96485.33212
SEED = 20260912
FEATURES = ["Ravg", "PhiTM", "DeltaXNa"]
FINAL_PARAMS = {"kernel": "rbf", "C": 3.0, "epsilon": 0.001, "gamma": "scale"}
NESTED_GRID = [
    {"kernel": "rbf", "C": c, "epsilon": eps, "gamma": gamma}
    for c in (1.0, 3.0, 10.0, 30.0)
    for eps in (0.001, 0.05, 0.1, 0.2)
    for gamma in ("scale", 0.1, 0.3, 0.5)
]
COLUMNS = {"formula": "Formula", "Ravg": "Ravg(Å)", "PhiTM": "ΦTM", "Qch": "Qch(mAh g−1)", "DeltaV": "ΔV(%)"}
TOLERANCE = 1e-12


def safe_number(expression: str) -> float:
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.UAdd: operator.pos, ast.USub: operator.neg}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](visit(node.operand))
        raise ValueError(expression)
    return float(visit(ast.parse(expression, mode="eval")))


def parse_formula(formula: str) -> dict[str, float]:
    clean = re.sub(r"[\s\[\]\(\)]", "", str(formula))
    out: dict[str, float] = {}
    for symbol, suffix in re.findall(r"([A-Z][a-z]?)([^A-Z]*)", clean):
        out[symbol] = out.get(symbol, 0.0) + (safe_number(suffix) if suffix else 1.0)
    return out


def load_training_data():
    """Return the 45 x 3 descriptor matrix, the signed ΔV target and the table rows."""
    frame = pd.read_excel(DATASET, sheet_name=0)
    missing = [column for column in COLUMNS.values() if column not in frame]
    if missing or len(frame) != 45 or frame[list(COLUMNS.values())].isna().any().any():
        raise ValueError("Expected the complete 45-row Supplementary Table 5; missing columns: %s" % missing)
    atomic = pd.read_csv(ELEMENTS).set_index("element")["atomic_weight"].astype(float).to_dict()
    delta_x_na = []
    for formula, qch in zip(frame[COLUMNS["formula"]], frame[COLUMNS["Qch"]]):
        composition = parse_formula(formula)
        mass = sum(atomic[element] * amount for element, amount in composition.items())
        delta_x_na.append(float(qch) * 3.6 * mass / FARADAY)
    x = np.column_stack([frame[COLUMNS["Ravg"]].to_numpy(float), frame[COLUMNS["PhiTM"]].to_numpy(float),
                         np.array(delta_x_na)])
    y = frame[COLUMNS["DeltaV"]].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid training matrix.")
    rows = frame.copy()
    rows["DeltaXNa_exact"] = x[:, 2]
    return x, y, rows


def metric_dict(y, pred) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "MAE": float(mean_absolute_error(y, pred)),
        "Pearson_r": float(pearsonr(y, pred).statistic),
        "n": int(len(y)),
    }


def delta_recipe(params: dict | None = None):
    return make_pipeline(StandardScaler(), SVR(**(params or FINAL_PARAMS)))


def repeated_oof(x, y, *, params: dict | None = None, repeats: int = 20, seed: int = SEED):
    """Mean out-of-fold prediction of each record over repeated 5-fold CV."""
    cv = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    sums = np.zeros(len(y), float)
    counts = np.zeros(len(y), int)
    recipe = delta_recipe(params)
    for train, test in cv.split(x):
        fitted = clone(recipe).fit(x[train], y[train])
        sums[test] += fitted.predict(x[test])
        counts[test] += 1
    if not np.all(counts == repeats):
        raise AssertionError("Repeated-CV coverage mismatch")
    return sums / counts


def nested_oof(x, y, repeats: int = 10):
    """Nested 5-fold CV: hyperparameters chosen by inner 5-fold RMSE within each outer fold."""
    sums = np.zeros(len(y), float)
    counts = np.zeros(len(y), int)
    selected: Counter = Counter()
    for repeat in range(repeats):
        outer = KFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for train, test in outer.split(x):
            inner = KFold(n_splits=5, shuffle=True, random_state=909 + repeat)
            best_params, best_rmse = None, float("inf")
            for params in NESTED_GRID:
                inner_pred = np.zeros(len(train), float)
                for inner_train, inner_test in inner.split(train):
                    fitted = delta_recipe(params).fit(x[train][inner_train], y[train][inner_train])
                    inner_pred[inner_test] = fitted.predict(x[train][inner_test])
                rmse = float(mean_squared_error(y[train], inner_pred) ** 0.5)
                if rmse < best_rmse:
                    best_rmse, best_params = rmse, params
            selected[json.dumps(best_params, sort_keys=True)] += 1
            fitted = delta_recipe(best_params).fit(x[train], y[train])
            sums[test] += fitted.predict(x[test])
            counts[test] += 1
    if not np.all(counts == repeats):
        raise AssertionError("Nested-CV coverage mismatch")
    return sums / counts, selected


def conformal_half_width(y, pred, coverage: float = 0.95):
    residuals = np.sort(np.abs(np.asarray(y) - np.asarray(pred)))
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * coverage)))
    return float(residuals[rank - 1]), int(rank)


def metric_differences(new: dict, reference: dict) -> dict:
    return {key: abs(float(new[key]) - float(reference[key])) for key in ("R2", "RMSE", "MAE", "Pearson_r")}


def compare_reference(fitted, x) -> dict:
    """Refitted model vs the model shipped with the application: predictions and fitted parameters."""
    with MODEL.open("rb") as handle:
        reference = pickle.load(handle)["model"]
    rng = np.random.default_rng(20261009)
    probe = np.vstack([x, rng.uniform(x.min(axis=0), x.max(axis=0), size=(100, x.shape[1]))])
    maximum = float(np.max(np.abs(fitted.predict(probe) - reference.predict(probe))))
    attributes = {}
    for step, fields in {"standardscaler": ["mean_", "scale_"],
                         "svr": ["support_", "support_vectors_", "dual_coef_", "intercept_"]}.items():
        for field in fields:
            left = np.asarray(getattr(fitted.named_steps[step], field))
            right = np.asarray(getattr(reference.named_steps[step], field))
            attributes["%s.%s" % (step, field)] = {
                "same_shape": left.shape == right.shape,
                "max_abs_difference": float(np.max(np.abs(left - right))) if left.shape == right.shape else None,
            }
    same = all(v["same_shape"] and v["max_abs_difference"] <= TOLERANCE for v in attributes.values())
    if maximum > TOLERANCE or not same:
        raise AssertionError("Refitted model differs from the shipped model: %s; %s" % (maximum, attributes))
    return {"passed": True, "max_abs_prediction_difference": maximum, "probe_count": len(probe),
            "attributes": attributes}


def train(output: Path, full_validation: bool = False) -> dict:
    output = output.resolve()
    if output == DATA_DIR.resolve() or DATA_DIR.resolve() in output.parents:
        raise ValueError("The output folder must not be inside data/.")
    if (output / MODEL.name).exists():
        raise FileExistsError("Choose a new output folder: %s" % output)
    x, y, rows = load_training_data()
    reference = json.loads(MODEL_META.read_text(encoding="utf-8"))["DeltaV"]

    fixed_oof = repeated_oof(x, y)
    fixed_metrics = metric_dict(y, fixed_oof)
    fixed_diff = metric_differences(fixed_metrics, reference["fixed_model_repeated_20x5_cv_metrics"])
    if max(fixed_diff.values()) > TOLERANCE:
        raise AssertionError("Repeated CV differs from model_meta.json: %s" % fixed_diff)
    fitted = delta_recipe().fit(x, y)
    apparent = metric_dict(y, fitted.predict(x))
    apparent_diff = metric_differences(apparent, reference["apparent_training_metrics"])
    if max(apparent_diff.values()) > TOLERANCE:
        raise AssertionError("Apparent-fit metrics differ from model_meta.json: %s" % apparent_diff)
    reference_check = compare_reference(fitted, x)
    half_width, rank = conformal_half_width(y, fixed_oof)
    interval = reference["prediction_interval"]
    if abs(half_width - interval["half_width"]) > TOLERANCE or rank != interval["calibration_rank"]:
        raise AssertionError("Reference interval differs from model_meta.json")

    report = {
        "dataset": {"file": DATASET.name, "sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
                    "rows": len(y)},
        "algorithm": {"pipeline": "StandardScaler + RBF-SVR", **FINAL_PARAMS, "features": FEATURES,
                      "target": "signed DeltaV (%)"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "fixed_cv": {"folds": 5, "repeats": 20, "seed": SEED,
                     "aggregation": "Metrics computed after averaging each record's 20 out-of-fold predictions.",
                     "metrics": fixed_metrics, "absolute_difference_from_model_meta": fixed_diff},
        "apparent_training_metrics": apparent,
        "reference_model_comparison": reference_check,
        "empirical_cv_reference_interval": {"half_width_percentage_points": half_width, "rank": rank, "n": len(y)},
        "nested_cv": {"executed": False},
    }
    rows["DeltaV_fit_raw"] = fitted.predict(x)
    rows["DeltaV_fit_output"] = np.minimum(rows["DeltaV_fit_raw"], -1e-6)
    rows["DeltaV_mean_oof_20x5"] = fixed_oof

    if full_validation:
        print("Running nested 10 x 5 cross-validation and seed sensitivity ...", flush=True)
        nested_pred, selected = nested_oof(x, y)
        nested_metrics = metric_dict(y, nested_pred)
        nested_diff = metric_differences(nested_metrics, reference["nested_model_selection_10x5_cv_metrics"])
        if max(nested_diff.values()) > TOLERANCE or dict(selected) != reference["nested_selected_parameter_counts"]:
            raise AssertionError("Nested CV differs from model_meta.json: %s" % nested_diff)
        seeds = {str(seed): metric_dict(y, repeated_oof(x, y, repeats=10, seed=seed)) for seed in (17, 42, 2026, SEED)}
        seed_diff = max(max(metric_differences(m, reference["fixed_model_seed_sensitivity_10x5"][s]).values())
                        for s, m in seeds.items())
        if seed_diff > TOLERANCE:
            raise AssertionError("Seed sensitivity differs from model_meta.json: %s" % seed_diff)
        report["nested_cv"] = {"executed": True, "metrics": nested_metrics, "parameter_selection_counts": dict(selected),
                               "absolute_difference_from_model_meta": nested_diff}
        report["seed_sensitivity_10x5"] = seeds
        report["seed_sensitivity_max_difference_from_model_meta"] = seed_diff
        rows["DeltaV_mean_oof_nested_10x5"] = nested_pred

    payload = {
        "features": FEATURES, "target": "DeltaV", "kind": "sklearn", "model": fitted,
        "best_params": {"pipeline": "StandardScaler + RBF-SVR", **FINAL_PARAMS, "literature_rows": 45,
                        "training_inputs": "literature_only"},
        "physics_constraints": {
            "signed_target": "DeltaV is fitted directly with its sign; absolute DeltaV is not the regression target.",
            "descriptor_interpretation": "Ravg, PhiTM and DeltaXNa are mechanism-linked descriptors; "
                                         "the fitted RBF model is not globally monotonic.",
        },
        "output_clip": {"max": -1e-6},
    }
    output.mkdir(parents=True, exist_ok=True)
    with (output / MODEL.name).open("wb") as handle:
        pickle.dump(payload, handle, protocol=4)
    rows.to_csv(output / "training_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "fixed_cv_metrics": fixed_metrics,
                      "max_prediction_difference_from_shipped_model": reference_check["max_abs_prediction_difference"],
                      "nested_cv_executed": full_validation, "all_checks_passed": True}, ensure_ascii=False, indent=2))
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "retrained")
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    train(args.output, args.full_validation)


if __name__ == "__main__":
    main()
