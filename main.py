from __future__ import annotations

import argparse
import ast
import html
import json
import math
import operator
import pickle
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline  # noqa: F401
from sklearn.preprocessing import StandardScaler  # noqa: F401
from sklearn.svm import SVR  # noqa: F401


APP_VERSION = "2026.9.24-manuscript-notation-sci-audit-r3"
PREDICT_BUTTON_LABEL = "Predict ΔV"
COMPARISON_Y_AXIS_LABEL = "Predicted ΔV (%)"
FARADAY = 96485.33212
DELTA_X_MODEL_DECIMALS = 2
QCH_DISPLAY_DECIMALS = 1
DELTA_V_DISPLAY_DECIMALS = 2
NA_LIMIT_TOLERANCE = 1e-12
FEATURE_DISPLAY_NAMES = {
    "XNa": "X_Na",
    "Ravg": "R_avg",
    "PhiTM": "Φ_TM",
    "DeltaXNa": "ΔX_Na",
    "radius_std": "radius_std",
    "phi_std": "phi_std",
    "mixing_entropy": "mixing_entropy",
    "n_tm_elements": "n_tm_elements",
}

# Rich-text versions are used only in the graphical interface.  Internal model
# keys remain unchanged so the verified model and its input order are preserved.
MATH_HTML = {
    "XNa": "<i>X</i><sub>Na</sub>",
    "Ravg": "<i>R</i><sub>avg</sub>",
    "PhiTM": "<i>Φ</i><sub>TM</sub>",
    "DeltaXNa": "Δ<i>X</i><sub>Na</sub>",
    "Qch": "<i>Q</i><sub>ch</sub>",
    "TMSum": "<i>TM</i><sub>sum</sub>",
    "nO": "<i>n</i><sub>O</sub>",
    "PhiCation": "<i>Φ</i><sub>cation</sub>",
    "PhiTMPhase": "<i>Φ</i><sub>TM</sub><sup>phase</sup>",
    "PhiNaPhase": "<i>Φ</i><sub>Na</sub><sup>phase</sup>",
    "PhiOPhase": "<i>Φ</i><sub>O</sub><sup>phase</sup>",
    "DeltaV": "Δ<i>V</i>",
}

# Zhao et al., Science 370, 708-711 (2020), DOI: 10.1126/science.aay9972.
# The published phase map uses ionic potentials in nm^-1 and the dividing line
# Φ_Na^phase = 0.4065 * Φ_cation + 1.572.
PHASE_RULE_SLOPE = 0.4065
PHASE_RULE_INTERCEPT = 1.572
PHASE_RULE_TM_SUM_MIN = 0.95
PHASE_RULE_TM_SUM_MAX = 1.05
PHASE_RULE_OXYGEN_TARGET = 2.0
PHASE_RULE_OXYGEN_TOLERANCE = 0.05
PHASE_RULE_CHARGE_TOLERANCE = 0.05
PHASE_RULE_BOUNDARY_EPSILON = 1e-9
PHASE_RULE_NEAR_BOUNDARY_MARGIN = 0.05


def round_half_up(value: float, decimals: int) -> float:
    quantum = Decimal("1").scaleb(-decimals)
    return float(Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_UP))


def canonical_delta_x_na(value: float) -> float:
    return round_half_up(value, DELTA_X_MODEL_DECIMALS)


def validate_delta_x_na(delta_x_na: float, x_na: float) -> float:
    """Validate the physical Na-extraction limit and return the model input.

    The raw value is checked before rounding so that an over-limit input cannot
    be hidden by the two-decimal model convention.  The rounded effective value
    is checked again because it is the value actually passed to the regressor.
    """
    raw_value = float(delta_x_na)
    initial_na = float(x_na)
    if not math.isfinite(raw_value):
        raise ValueError("ΔX_Na must be a finite number.")
    if raw_value < 0:
        raise ValueError(
            f"ΔX_Na={raw_value:.6g} is invalid. Required range: "
            f"0 ≤ ΔX_Na ≤ X_Na ({initial_na:.6g}). Prediction was blocked."
        )
    if raw_value > initial_na + NA_LIMIT_TOLERANCE:
        raise ValueError(
            f"ΔX_Na={raw_value:.6g} exceeds the initial Na stoichiometry "
            f"X_Na={initial_na:.6g}. Required range: 0 ≤ ΔX_Na ≤ X_Na. "
            "Prediction was blocked."
        )
    effective_value = canonical_delta_x_na(raw_value)
    if effective_value > initial_na + NA_LIMIT_TOLERANCE:
        raise ValueError(
            f"The two-decimal Effective ΔX_Na={effective_value:.2f} exceeds "
            f"X_Na={initial_na:.6g}. Reduce ΔX_Na (or Q_ch) so that the value "
            "used by the model satisfies 0 ≤ ΔX_Na ≤ X_Na. Prediction was blocked."
        )
    return effective_value


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


DATA_DIR = resource_root() / "data"
ASSETS_DIR = resource_root() / "assets"
APP_ICON_PATH = ASSETS_DIR / "EM_VRF_icon.ico"


@dataclass(frozen=True)
class Element:
    symbol: str
    valence: float
    radius: float
    is_tm: bool
    atomic_weight: float


def load_elements() -> dict[str, Element]:
    table = pd.read_csv(DATA_DIR / "element_database.csv")
    return {
        row.element: Element(
            str(row.element), float(row.default_valence), float(row.ionic_radius),
            str(row.is_tm_site).lower() == "true", float(row.atomic_weight)
        )
        for row in table.itertuples(index=False)
    }


ELEMENTS = load_elements()


def _safe_number(expression: str) -> float:
    operators = {
        ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.UAdd: operator.pos, ast.USub: operator.neg,
    }
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](visit(node.operand))
        raise ValueError(f"Unsupported coefficient expression: {expression}")

    result = float(visit(tree))
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Coefficient must be a non-negative finite number: {expression}")
    return result


def parse_formula(formula: str) -> dict[str, float]:
    translation = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    cleaned = formula.translate(translation)
    cleaned = re.sub(r"[\s\[\]\(\)]", "", cleaned)
    if not cleaned:
        raise ValueError("Formula cannot be empty.")
    matches = list(re.finditer(r"([A-Z][a-z]?)([^A-Z]*)", cleaned))
    if not matches or matches[0].start() != 0 or matches[-1].end() != len(cleaned):
        raise ValueError(
            "Unable to parse the formula. Use element symbols followed by decimals "
            "or simple fractions."
        )
    composition: dict[str, float] = {}
    for match in matches:
        symbol = match.group(1)
        suffix = match.group(2)
        if symbol not in ELEMENTS:
            raise ValueError(f"Element {symbol} is not included in the application database.")
        coefficient = 1.0 if suffix == "" else _safe_number(suffix)
        composition[symbol] = composition.get(symbol, 0.0) + coefficient
    return composition


def formula_string(composition: dict[str, float]) -> str:
    order = ["Na"] + [s for s, e in ELEMENTS.items() if e.is_tm] + ["O"]
    pieces = []
    for symbol in order:
        value = composition.get(symbol, 0.0)
        if value <= 1e-12:
            continue
        suffix = "" if abs(value - 1.0) < 1e-10 else f"{value:.6g}"
        pieces.append(f"{symbol}{suffix}")
    return "".join(pieces)


def normalize_input_column_name(name: object) -> str:
    """Normalize article-style and legacy batch headers for matching."""
    value = str(name).strip().lower()
    return re.sub(r"[\s_\-()/%⁻¹]+", "", value)


def standardize_display_symbols(value):
    """Return a display-only copy using the manuscript notation."""
    if isinstance(value, dict):
        return {
            standardize_display_symbols(key): standardize_display_symbols(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [standardize_display_symbols(item) for item in value]
    if isinstance(value, tuple):
        return tuple(standardize_display_symbols(item) for item in value)
    if not isinstance(value, str):
        return value
    text = value
    for old, new in (
        ("DeltaXNa", "ΔX_Na"),
        ("DeltaV", "ΔV"),
        ("PhiCation", "Φ_cation"),
        ("PhiTM", "Φ_TM"),
        ("PhiNa", "Φ_Na^phase"),
        ("PhiO", "Φ_O^phase"),
        ("Ravg", "R_avg"),
        ("XNa", "X_Na"),
        ("Qch", "Q_ch"),
    ):
        text = text.replace(old, new)
    return text


def calculate_descriptors(composition: dict[str, float]) -> dict[str, float]:
    tm = [(symbol, amount) for symbol, amount in composition.items() if ELEMENTS[symbol].is_tm]
    tm_sum = sum(amount for _, amount in tm)
    if tm_sum <= 0:
        raise ValueError("The formula must contain at least one TM-site element.")
    ravg = sum(amount * ELEMENTS[symbol].radius for symbol, amount in tm) / tm_sum
    phi_tm = sum(
        amount * ELEMENTS[symbol].valence / ELEMENTS[symbol].radius
        for symbol, amount in tm
    ) / tm_sum
    fractions = np.array([amount / tm_sum for _, amount in tm], dtype=float)
    radii = np.array([ELEMENTS[symbol].radius for symbol, _ in tm], dtype=float)
    potentials = np.array([
        ELEMENTS[symbol].valence / ELEMENTS[symbol].radius for symbol, _ in tm
    ], dtype=float)
    radius_std = float(np.sqrt(np.sum(fractions * (radii - ravg) ** 2)))
    phi_std = float(np.sqrt(np.sum(fractions * (potentials - phi_tm) ** 2)))
    mixing_entropy = float(-np.sum(fractions * np.log(fractions)))
    mass = sum(amount * ELEMENTS[symbol].atomic_weight for symbol, amount in composition.items())
    total_charge = sum(amount * ELEMENTS[symbol].valence for symbol, amount in composition.items())
    return {
        "XNa": float(composition.get("Na", 0.0)),
        "Ravg": float(ravg),
        "PhiTM": float(phi_tm),
        "radius_std": radius_std,
        "phi_std": phi_std,
        "mixing_entropy": mixing_entropy,
        "n_tm_elements": int(len(tm)),
        "TM_sum": float(tm_sum),
        "molar_mass": float(mass),
        "charge_gap": float(total_charge),
        "oxygen": float(composition.get("O", 0.0)),
    }


def assess_stacking_phase(
    composition: dict[str, float],
    descriptors: dict[str, float] | None = None,
) -> dict[str, object]:
    """Apply the Zhao cationic-potential P2/O3 composition rule.

    This is a composition-screening rule, not a guarantee that a single O3
    phase is synthesizable.  Values are calculated in nm^-1 to match the
    published phase map.  The element table stores radii in angstroms.
    """
    desc = descriptors or calculate_descriptors(composition)
    reasons: list[str] = []
    if desc["XNa"] <= 0:
        reasons.append("The Zhao rule requires a positive Na-layer content.")
    if not PHASE_RULE_TM_SUM_MIN <= desc["TM_sum"] <= PHASE_RULE_TM_SUM_MAX:
        reasons.append(
            f"TM-site coefficient sum {desc['TM_sum']:.4f} is outside "
            f"[{PHASE_RULE_TM_SUM_MIN:.2f}, {PHASE_RULE_TM_SUM_MAX:.2f}]."
        )
    if abs(desc["oxygen"] - PHASE_RULE_OXYGEN_TARGET) > PHASE_RULE_OXYGEN_TOLERANCE:
        reasons.append(
            f"Oxygen coefficient {desc['oxygen']:.4f} is not compatible with "
            f"the Na_x(TM)O2 scope of the rule."
        )
    if abs(desc["charge_gap"]) > PHASE_RULE_CHARGE_TOLERANCE:
        reasons.append(
            "Default oxidation states do not satisfy charge balance "
            f"(gap {desc['charge_gap']:+.3f}; allowed magnitude "
            f"{PHASE_RULE_CHARGE_TOLERANCE:.2f})."
        )

    base = {
        "phase_rule": "Φ_Na^phase > 0.4065 * Φ_cation + 1.572",
        "phase_rule_reference": "Zhao et al., Science 370, 708-711 (2020)",
        "phase_rule_scope": "composition-based P2/O3 stacking preference",
        "phase_gate_reasons": reasons,
    }
    if reasons:
        return {
            **base,
            "phase_label": "Not assessable",
            "phase_code": "NOT_ASSESSABLE",
            "prediction_allowed": False,
            "PhiTM_phase_nm_inv": None,
            "PhiNa_phase_nm_inv": None,
            "PhiO_phase_nm_inv": None,
            "PhiCation_phase_nm_inv": None,
            "phase_boundary_phi_na": None,
            "phase_margin": None,
            "near_phase_boundary": False,
        }

    phi_tm_nm = 10.0 * sum(
        amount * ELEMENTS[symbol].valence / ELEMENTS[symbol].radius
        for symbol, amount in composition.items()
        if ELEMENTS[symbol].is_tm
    )
    phi_na_nm = (
        10.0
        * desc["XNa"]
        * abs(ELEMENTS["Na"].valence)
        / ELEMENTS["Na"].radius
    )
    phi_o_nm = (
        10.0
        * PHASE_RULE_OXYGEN_TARGET
        * abs(ELEMENTS["O"].valence)
        / ELEMENTS["O"].radius
    )
    phi_cation_nm = phi_tm_nm * phi_na_nm / phi_o_nm
    boundary = PHASE_RULE_SLOPE * phi_cation_nm + PHASE_RULE_INTERCEPT
    margin = phi_na_nm - boundary
    near_boundary = abs(margin) < PHASE_RULE_NEAR_BOUNDARY_MARGIN
    if margin > PHASE_RULE_BOUNDARY_EPSILON:
        phase_label = "O3-favored"
        phase_code = "O3"
        prediction_allowed = True
    elif margin < -PHASE_RULE_BOUNDARY_EPSILON:
        phase_label = "P2-favored"
        phase_code = "P2"
        prediction_allowed = False
    else:
        phase_label = "Boundary / uncertain"
        phase_code = "BOUNDARY"
        prediction_allowed = False
    return {
        **base,
        "phase_label": phase_label,
        "phase_code": phase_code,
        "prediction_allowed": prediction_allowed,
        "PhiTM_phase_nm_inv": float(phi_tm_nm),
        "PhiNa_phase_nm_inv": float(phi_na_nm),
        "PhiO_phase_nm_inv": float(phi_o_nm),
        "PhiCation_phase_nm_inv": float(phi_cation_nm),
        "phase_boundary_phi_na": float(boundary),
        "phase_margin": float(margin),
        "near_phase_boundary": bool(near_boundary),
    }


def load_models():
    with (DATA_DIR / "EM-VRF-ΔV.pkl").open("rb") as handle:
        dv = pickle.load(handle)
    meta = json.loads((DATA_DIR / "model_meta.json").read_text(encoding="utf-8"))
    domain = json.loads((DATA_DIR / "training_domain.json").read_text(encoding="utf-8"))
    return dv, meta, domain


DV_MODEL, MODEL_META, TRAINING_DOMAIN = load_models()


def predict_signed_delta_v(feature_values: dict[str, float]) -> float:
    x = np.array([[feature_values[name] for name in DV_MODEL["features"]]], dtype=float)
    if DV_MODEL.get("kind") == "scaled_bounded_ridge":
        center = np.asarray(DV_MODEL["center"], dtype=float)
        scale = np.asarray(DV_MODEL["scale"], dtype=float)
        coefficients = np.asarray(DV_MODEL["coef"], dtype=float)
        raw_value = float(
            DV_MODEL["intercept"] + (((x - center) / scale) @ coefficients)[0]
        )
    else:
        raw_value = float(DV_MODEL["model"].predict(x)[0])
    return min(raw_value, float(DV_MODEL.get("output_clip", {}).get("max", -1e-6)))


def predict_from_composition(composition: dict[str, float], delta_x_na: float) -> dict[str, object]:
    desc = calculate_descriptors(composition)
    delta_x_na = validate_delta_x_na(delta_x_na, desc["XNa"])
    warnings: list[str] = []
    if not 0.95 <= desc["TM_sum"] <= 1.05:
        warnings.append(f"TM-site coefficient sum is {desc['TM_sum']:.4f}, which deviates from 1.")
    if abs(desc["charge_gap"]) > 0.25:
        warnings.append(
            f"Charge-balance gap estimated from default valences is "
            f"{desc['charge_gap']:+.3f}."
        )
    phase = assess_stacking_phase(composition, desc)
    if phase["near_phase_boundary"]:
        warnings.append(
            "The composition lies within 0.05 of the Zhao P2/O3 dividing line; "
            "the phase preference is sensitive to ionic-state and radius assumptions."
        )
    base_result = {
        "formula": formula_string(composition),
        **desc,
        "DeltaXNa": float(delta_x_na),
        **phase,
        "prediction_target": "ΔV",
        "warnings": warnings,
    }
    if not phase["prediction_allowed"]:
        return {
            **base_result,
            "DeltaV_pred": None,
            "DeltaV_lower": None,
            "DeltaV_upper": None,
            "risk": None,
        }

    feature_values = {**desc, "DeltaXNa": float(delta_x_na)}
    delta_v = predict_signed_delta_v(feature_values)
    check_values = {
        "XNa": desc["XNa"], "Ravg": desc["Ravg"], "PhiTM": desc["PhiTM"],
        "DeltaXNa": delta_x_na,
        "radius_std": desc["radius_std"], "phi_std": desc["phi_std"],
        "mixing_entropy": desc["mixing_entropy"],
        "n_tm_elements": desc["n_tm_elements"],
    }
    for key, value in check_values.items():
        bounds = TRAINING_DOMAIN[key]
        if value < bounds["min"] or value > bounds["max"]:
            display_name = FEATURE_DISPLAY_NAMES.get(key, key)
            warnings.append(
                f"{display_name}={value:.4f} is outside the training range "
                f"[{bounds['min']:.4f}, {bounds['max']:.4f}]."
            )
    response_magnitude = abs(delta_v)
    risk = "Low" if response_magnitude <= 1.0 else (
        "Medium" if response_magnitude <= 2.5 else "High"
    )
    dv_half_width = float(MODEL_META["DeltaV"]["prediction_interval"]["half_width"])
    return {
        **base_result,
        "DeltaV_pred": delta_v,
        "DeltaV_lower": delta_v - dv_half_width,
        "DeltaV_upper": delta_v + dv_half_width,
        "risk": risk,
    }


def predict_formula(formula: str, delta_x_na: float | None = None, qch: float | None = None):
    composition = parse_formula(formula)
    if delta_x_na is None:
        if qch is None:
            raise ValueError("Either ΔX_Na or Q_ch must be provided.")
        mass = calculate_descriptors(composition)["molar_mass"]
        delta_x_na = float(qch) * mass * 3.6 / FARADAY
    return predict_from_composition(composition, float(delta_x_na))


def prediction_text(result: dict[str, object]) -> str:
    warning_lines = "\n".join(f"  • {item}" for item in result["warnings"]) or "  None"
    phase_reason_lines = (
        "\n".join(f"  • {item}" for item in result["phase_gate_reasons"])
        or "  None"
    )
    phase_values = ""
    if result["PhiCation_phase_nm_inv"] is not None:
        phase_values = (
            f"\nΦ_cation = {result['PhiCation_phase_nm_inv']:.6f} nm⁻¹"
            f"\nΦ_Na^phase = {result['PhiNa_phase_nm_inv']:.6f} nm⁻¹"
            f"\nBoundary Φ_Na^phase = {result['phase_boundary_phi_na']:.6f} nm⁻¹"
            f"\nPhase margin m = {result['phase_margin']:+.6f} nm⁻¹"
        )
    header = (
        f"Formula: {result['formula']}\n"
        f"X_Na = {result['XNa']:.6f}\n"
        f"R_avg = {result['Ravg']:.6f} Å\n"
        f"Φ_TM = {result['PhiTM']:.6f} Å⁻¹\n"
        f"Effective ΔX_Na = {result['DeltaXNa']:.2f}\n\n"
        f"Zhao criterion: {result['phase_label']}"
        f"{phase_values}\n"
        "Scope note: this composition rule screens P2/O3 stacking preference and does "
        "not guarantee synthesis of a single O3 phase.\n"
    )
    if not result["prediction_allowed"]:
        return (
            header
            + "\nΔV = Not predicted (only O3-favored candidates pass the phase gate).\n"
            + f"\nPhase-gate reasons:\n{phase_reason_lines}"
            + f"\n\nWarnings:\n{warning_lines}"
        )
    return (
        header
        + f"\nΔV = {result['DeltaV_pred']:+.2f}%  "
        + f"(CV-based empirical reference interval: {result['DeltaV_lower']:.2f}%–"
        + f"{result['DeltaV_upper']:.2f}%)\n"
        + "Note: signed ΔV is retained; a negative value indicates volume contraction "
        + "relative to the initial unit cell during charging.\n"
        + "The interval is not an independently validated 95% coverage claim.\n"
        + f"Volume-response risk: {result['risk']}\n"
        + f"\nPhase-gate reasons:\n{phase_reason_lines}"
        + f"\n\nWarnings:\n{warning_lines}"
    )


def public_prediction_payload(result: dict[str, object]) -> dict[str, object]:
    """Return the documented, manuscript-facing prediction schema.

    Internal dictionary keys are deliberately not exposed by the command-line
    interface.  They remain unchanged inside the application to preserve model
    compatibility.
    """
    payload = {
        "Formula": result["formula"],
        "X_Na": result["XNa"],
        "TM_sum": result["TM_sum"],
        "R_avg (Å)": result["Ravg"],
        "Φ_TM (Å⁻¹)": result["PhiTM"],
        "Phase": result["phase_label"],
        "Φ_cation (nm⁻¹)": result["PhiCation_phase_nm_inv"],
        "Φ_Na^phase (nm⁻¹)": result["PhiNa_phase_nm_inv"],
        "Phase-boundary Φ_Na^phase (nm⁻¹)": result["phase_boundary_phi_na"],
        "Phase margin m (nm⁻¹)": result["phase_margin"],
        "Prediction allowed": bool(result["prediction_allowed"]),
        "Effective ΔX_Na": result["DeltaXNa"],
        "Predicted ΔV (%)": result["DeltaV_pred"],
        "CV-based empirical interval lower (%)": result["DeltaV_lower"],
        "CV-based empirical interval upper (%)": result["DeltaV_upper"],
        "Heuristic |ΔV| band": result["risk"],
        "Warnings": list(result["warnings"]),
        "Phase-gate reasons": list(result["phase_gate_reasons"]),
    }
    return payload


def self_test() -> dict[str, object]:
    o3_reference = predict_formula(
        "NaLi1/3Ti1/6Mn1/2O2", delta_x_na=0.50
    )
    p2_reference = predict_formula(
        "Na5/6Li5/18Mn13/18O2", delta_x_na=0.50
    )
    near_o3_reference = predict_formula(
        "Na2/3Mg1/3Ti2/3O2", delta_x_na=0.50
    )
    unresolved_reference = predict_formula("NaRuO2", delta_x_na=0.50)
    assert o3_reference["phase_code"] == "O3"
    assert o3_reference["prediction_allowed"] is True
    assert o3_reference["DeltaV_pred"] is not None
    assert abs(o3_reference["PhiCation_phase_nm_inv"] - 18.2347) < 5e-4
    assert p2_reference["phase_code"] == "P2"
    assert p2_reference["prediction_allowed"] is False
    assert p2_reference["DeltaV_pred"] is None
    assert abs(p2_reference["PhiCation_phase_nm_inv"] - 16.6314) < 5e-4
    assert near_o3_reference["phase_code"] == "O3"
    assert near_o3_reference["near_phase_boundary"] is True
    assert unresolved_reference["phase_code"] == "NOT_ASSESSABLE"
    assert unresolved_reference["DeltaV_pred"] is None

    anchor_inputs = {
        "NMTS": (
            "Na0.92Ni0.4Mn0.4Al0.05Ti0.1Li0.03Sb0.02O2", 149.2,
            -1.4125170757941106,
        ),
        "NMTS-A": (
            "Na0.92Ni0.4Mn0.4Al0.08Ti0.08Li0.02Sb0.02O2", 139.2,
            -0.5285779944846007,
        ),
        "NMTS-M": (
            "Na0.905Ni0.4Mn0.4Al0.05Ti0.1Li0.015Mg0.02Mo0.015O2", 140.6,
            -0.6258592946789454,
        ),
        "NMTS-L": (
            "NaNi0.4Mn0.4Al0.06Ti0.04Li0.06Sb0.04O2", 150.2,
            -1.9338138998271188,
        ),
    }
    anchor_results = {}
    for name, (formula, qch, expected_delta_v) in anchor_inputs.items():
        result = predict_formula(formula, qch=qch)
        assert result["phase_code"] == "O3"
        assert result["prediction_allowed"] is True
        assert abs(result["DeltaV_pred"] - expected_delta_v) < 1e-12
        anchor_results[name] = {
            "phase": result["phase_label"],
            "Φ_cation (nm⁻¹)": result["PhiCation_phase_nm_inv"],
            "Φ_Na^phase (nm⁻¹)": result["PhiNa_phase_nm_inv"],
            "phase margin m (nm⁻¹)": result["phase_margin"],
            "Effective ΔX_Na": result["DeltaXNa"],
            "ΔV (%)": result["DeltaV_pred"],
        }

    precision_formula = "Na0.92Ni0.4Mn0.4Al0.05Ti0.1Li0.03Sb0.02O2"
    direct = predict_formula(precision_formula, delta_x_na=0.60)
    capacity = predict_formula(precision_formula, qch=149.2)
    assert direct["DeltaXNa"] == capacity["DeltaXNa"] == 0.60
    assert direct["DeltaV_pred"] == capacity["DeltaV_pred"]

    def expect_na_limit_block(*, formula, delta_x_na=None, qch=None):
        try:
            predict_formula(formula, delta_x_na=delta_x_na, qch=qch)
        except ValueError as exc:
            message = str(exc)
            assert "0 ≤ ΔX_Na ≤ X_Na" in message
            assert "Prediction was blocked" in message
            return message
        raise AssertionError("Expected the Na-extraction limit to block prediction.")

    assert predict_formula(precision_formula, delta_x_na=0.92)["DeltaXNa"] == 0.92
    raw_over_limit_message = expect_na_limit_block(
        formula=precision_formula, delta_x_na=0.920001
    )
    capacity_over_limit_message = expect_na_limit_block(
        formula=precision_formula, qch=229.7
    )
    rounded_over_limit_message = expect_na_limit_block(
        formula="Na0.905Ni0.4Mn0.4Al0.05Ti0.1Li0.015Mg0.02Mo0.015O2",
        delta_x_na=0.905,
    )
    p2_over_limit_message = expect_na_limit_block(
        formula="Na5/6Li5/18Mn13/18O2", delta_x_na=0.84
    )
    return {
        "status": "ok",
        "version": APP_VERSION,
        "phase_gate_tests": {
            "O3_reference": {
                "formula": o3_reference["formula"],
                "phase": o3_reference["phase_label"],
                "phase margin m (nm⁻¹)": o3_reference["phase_margin"],
            },
            "P2_reference": {
                "formula": p2_reference["formula"],
                "phase": p2_reference["phase_label"],
                "phase margin m (nm⁻¹)": p2_reference["phase_margin"],
                "ΔV (%)": p2_reference["DeltaV_pred"],
            },
            "not_assessable_reference": {
                "formula": unresolved_reference["formula"],
                "phase": unresolved_reference["phase_label"],
                "reasons": unresolved_reference["phase_gate_reasons"],
            },
        },
        "anchor_regression": anchor_results,
        "precision_equivalence": {
            "ΔX_Na input": 0.60,
            "Q_ch input (mAh g⁻¹)": 149.2,
            "Effective ΔX_Na": direct["DeltaXNa"],
            "ΔV display": f"{direct['DeltaV_pred']:+.2f}%",
        },
        "Na_extraction_guard": {
            "rule": "0 ≤ ΔX_Na ≤ X_Na",
            "direct_raw_over_limit": raw_over_limit_message,
            "capacity_over_limit": capacity_over_limit_message,
            "rounded_effective_over_limit": rounded_over_limit_message,
            "precedes_phase_gate": p2_over_limit_message,
        },
    }


def run_gui(smoke_test: bool = False, snapshot_path: str | None = None) -> int:
    from PySide6.QtCore import QPointF, QRectF, QSignalBlocker, Qt, Signal
    from PySide6.QtGui import (
        QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QTextDocument,
    )
    from PySide6.QtWidgets import (
        QApplication, QButtonGroup, QCheckBox, QDoubleSpinBox, QFileDialog,
        QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QHeaderView, QMainWindow, QMessageBox,
        QPushButton, QRadioButton, QScrollArea, QSplitter, QStackedWidget, QTabWidget,
        QStyle, QStyleOptionHeader, QTableWidget, QTableWidgetItem, QTextEdit,
        QVBoxLayout, QWidget,
    )

    def rich_label(markup: str, suffix: str = "") -> QLabel:
        label = QLabel(f"{markup}{html.escape(suffix)}")
        label.setTextFormat(Qt.TextFormat.RichText)
        return label

    def manuscript_html_text(value: object) -> str:
        rendered = html.escape(str(value))
        for plain, markup in (
            ("Φ_TM^phase", MATH_HTML["PhiTMPhase"]),
            ("Φ_Na^phase", MATH_HTML["PhiNaPhase"]),
            ("Φ_O^phase", MATH_HTML["PhiOPhase"]),
            ("Φ_cation", MATH_HTML["PhiCation"]),
            ("ΔX_Na", MATH_HTML["DeltaXNa"]),
            ("Φ_TM", MATH_HTML["PhiTM"]),
            ("R_avg", MATH_HTML["Ravg"]),
            ("TM_sum", MATH_HTML["TMSum"]),
            ("X_Na", MATH_HTML["XNa"]),
            ("Q_ch", MATH_HTML["Qch"]),
            ("n_O", MATH_HTML["nO"]),
        ):
            rendered = rendered.replace(plain, markup)
        return rendered

    def rich_radio(prefix: str, symbol_markup: str, suffix: str = ""):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        radio = QRadioButton(prefix)
        symbol = rich_label(symbol_markup, suffix)
        layout.addWidget(radio)
        layout.addWidget(symbol)
        layout.addStretch()
        return radio, container

    def rich_column_header(column: str) -> str:
        exact = {
            "R_avg (Å)": f"{MATH_HTML['Ravg']} (Å)",
            "Φ_TM (Å⁻¹)": f"{MATH_HTML['PhiTM']} (Å<sup>−1</sup>)",
            "Φ_cation (nm⁻¹)": f"{MATH_HTML['PhiCation']} (nm<sup>−1</sup>)",
            "Φ_Na^phase (nm⁻¹)": f"{MATH_HTML['PhiNaPhase']} (nm<sup>−1</sup>)",
            "Boundary Φ_Na^phase (nm⁻¹)": (
                f"Boundary {MATH_HTML['PhiNaPhase']} (nm<sup>−1</sup>)"
            ),
            "Phase margin m (nm⁻¹)": "Phase margin <i>m</i> (nm<sup>−1</sup>)",
            "Effective ΔX_Na": f"Model input {MATH_HTML['DeltaXNa']}",
            "ΔX_Na": MATH_HTML["DeltaXNa"],
            "Predicted ΔV (%)": f"Predicted {MATH_HTML['DeltaV']} (%)",
        }
        return exact.get(column, html.escape(str(column)))

    class RichTextHeaderView(QHeaderView):
        """Paint scientific symbols with real HTML subscripts/superscripts."""

        def __init__(self, orientation, parent=None):
            super().__init__(orientation, parent)
            self._rich_labels: list[str] = []
            self.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setFixedHeight(38)

        def set_rich_labels(self, labels):
            self._rich_labels = [str(label) for label in labels]
            self.viewport().update()

        def paintSection(self, painter, rect, logical_index):
            if not rect.isValid():
                return
            painter.save()
            option = QStyleOptionHeader()
            self.initStyleOption(option)
            option.rect = rect
            option.section = logical_index
            option.text = ""
            self.style().drawControl(
                QStyle.ControlElement.CE_Header, option, painter, self
            )
            if 0 <= logical_index < len(self._rich_labels):
                markup = self._rich_labels[logical_index]
            else:
                value = self.model().headerData(
                    logical_index, self.orientation(), Qt.ItemDataRole.DisplayRole
                )
                markup = html.escape("" if value is None else str(value))
            document = QTextDocument()
            document.setDefaultFont(self.font())
            document.setDocumentMargin(0)
            document.setHtml(f"<div align='center'>{markup}</div>")
            document.setTextWidth(max(1.0, rect.width() - 8.0))
            height = document.size().height()
            painter.translate(
                rect.left() + 4.0,
                rect.top() + max(0.0, (rect.height() - height) / 2.0),
            )
            document.drawContents(painter)
            painter.restore()

    def positive_charge(composition: dict[str, float]) -> float:
        return sum(
            amount * ELEMENTS[symbol].valence
            for symbol, amount in composition.items()
            if symbol != "O" and ELEMENTS[symbol].valence > 0
        )

    class DeltaVComparisonChart(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.items = []
            self.y_axis_label = COMPARISON_Y_AXIS_LABEL
            self.setMinimumHeight(330)
            self.setMinimumWidth(380)

        def set_items(self, items):
            self.items = list(items)
            self.setMinimumWidth(max(380, 85 * len(self.items) + 70))
            self.update()

        @staticmethod
        def _blend(start: QColor, end: QColor, fraction: float) -> QColor:
            fraction = max(0.0, min(1.0, fraction))
            return QColor(
                round(start.red() + (end.red() - start.red()) * fraction),
                round(start.green() + (end.green() - start.green()) * fraction),
                round(start.blue() + (end.blue() - start.blue()) * fraction),
            )

        def paintEvent(self, event):
            del event
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#F4F6F8"))

            outer = QRectF(self.rect()).adjusted(12, 12, -12, -12)
            painter.setPen(QPen(QColor("#D9DEE5"), 1))
            painter.drawRoundedRect(outer, 6, 6)
            if not self.items:
                painter.setPen(QColor("#6B7280"))
                painter.drawText(
                    outer,
                    Qt.AlignmentFlag.AlignCenter,
                    "Run two or more predictions to compare signed ΔV",
                )
                return

            plot = outer.adjusted(76, 24, -24, -62)
            values = [float(item["DeltaV_pred"]) for item in self.items]
            lower_values = [float(item["DeltaV_lower"]) for item in self.items]
            upper_values = [float(item["DeltaV_upper"]) for item in self.items]
            data_min = min(values + lower_values)
            data_max = max(values + upper_values)
            y_min = min(data_min, 0.0)
            y_max = max(data_max, 0.0)
            span = max(y_max - y_min, 0.5)
            y_min -= 0.08 * span
            y_max += 0.08 * span

            def y_at(value):
                return plot.bottom() - (value - y_min) / (y_max - y_min) * plot.height()

            count = len(self.items)
            x_step = plot.width() / max(count - 1, 1)
            x_values = [
                plot.center().x() if count == 1 else plot.left() + index * x_step
                for index in range(count)
            ]

            painter.setFont(self.font())
            for tick in range(6):
                value = y_min + (y_max - y_min) * tick / 5
                y = y_at(value)
                painter.setPen(QPen(QColor("#D9DEE5"), 1, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
                painter.setPen(QColor("#626A73"))
                painter.drawText(
                    QRectF(outer.left(), y - 10, 64, 20),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    f"{value:.1f}",
                )

            painter.setPen(QPen(QColor("#5F6770"), 2))
            painter.drawLine(plot.bottomLeft(), plot.bottomRight())
            painter.drawLine(plot.bottomLeft(), plot.topLeft())
            zero_y = y_at(0.0)
            painter.setPen(QPen(QColor("#9AA1A9"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(plot.left(), zero_y), QPointF(plot.right(), zero_y))

            if count > 1:
                path = QPainterPath(QPointF(x_values[0], y_at(values[0])))
                for index in range(1, count):
                    previous_x = x_values[index - 1]
                    current_x = x_values[index]
                    midpoint = (previous_x + current_x) / 2
                    path.cubicTo(
                        QPointF(midpoint, y_at(values[index - 1])),
                        QPointF(midpoint, y_at(values[index])),
                        QPointF(current_x, y_at(values[index])),
                    )
                painter.setPen(QPen(QColor("#4F75A3"), 2.5))
                painter.drawPath(path)

            low_color = QColor("#436388")
            high_color = QColor("#9F5E58")
            value_span = max(max(values) - min(values), 1e-12)
            for index, item in enumerate(self.items):
                x = x_values[index]
                value = values[index]
                lower_y = y_at(float(item["DeltaV_lower"]))
                upper_y = y_at(float(item["DeltaV_upper"]))
                painter.setPen(QPen(QColor("#98A0A8"), 1.2))
                painter.drawLine(QPointF(x, lower_y), QPointF(x, upper_y))
                painter.drawLine(QPointF(x - 5, lower_y), QPointF(x + 5, lower_y))
                painter.drawLine(QPointF(x - 5, upper_y), QPointF(x + 5, upper_y))

                fraction = (value - min(values)) / value_span
                point_color = self._blend(low_color, high_color, fraction)
                painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
                painter.setBrush(point_color)
                painter.drawEllipse(QPointF(x, y_at(value)), 7, 7)
                painter.setBrush(Qt.BrushStyle.NoBrush)

                painter.setPen(QColor("#333940"))
                painter.drawText(
                    QRectF(x - 38, y_at(value) - 28, 76, 18),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{value:+.2f}%",
                )
                painter.drawText(
                    QRectF(x - 45, plot.bottom() + 8, 90, 20),
                    Qt.AlignmentFlag.AlignCenter,
                    str(item["candidate"]),
                )

            painter.setPen(QColor("#555D66"))
            painter.drawText(
                QRectF(plot.left(), outer.bottom() - 30, plot.width(), 24),
                Qt.AlignmentFlag.AlignCenter,
                "Candidate compositions",
            )
            painter.save()
            painter.translate(outer.left() + 16, plot.center().y())
            painter.rotate(-90)
            painter.drawText(
                QRectF(-plot.height() / 2, -12, plot.height(), 24),
                Qt.AlignmentFlag.AlignCenter,
                self.y_axis_label,
            )
            painter.restore()

    class PeriodicSelector(QWidget):
        selectionChanged = Signal(list)

        def __init__(self, allowed_elements, parent=None):
            super().__init__(parent)
            self.allowed_elements = set(allowed_elements)
            self.checkboxes = {}
            layout = QVBoxLayout(self)
            box = QGroupBox("Selectable dopant elements")
            grid = QGridLayout(box)
            periodic_layout = [
                ("Li", "", "", "", "", "", "", "", "", "", "", "", "B", "", "", "", "", ""),
                ("", "Mg", "", "", "", "", "", "", "", "", "", "", "Al", "", "", "", "", ""),
                ("", "", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "", "", "", "", ""),
                ("", "", "Y", "Zr", "Nb", "Mo", "", "Ru", "", "", "", "", "Sn", "Sb", "", "", "", ""),
                ("", "", "Ce", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""),
            ]
            for row, values in enumerate(periodic_layout):
                for column, symbol in enumerate(values):
                    if not symbol:
                        continue
                    check = QCheckBox(symbol)
                    check.setEnabled(symbol in self.allowed_elements)
                    check.toggled.connect(self._emit)
                    grid.addWidget(check, row, column)
                    self.checkboxes[symbol] = check
            for symbol in ("Ni", "Mn"):
                if symbol in self.checkboxes:
                    self.checkboxes[symbol].setChecked(True)
            layout.addWidget(box)

        def selected_elements(self):
            return [symbol for symbol, box in self.checkboxes.items() if box.isChecked()]

        def _emit(self):
            self.selectionChanged.emit(self.selected_elements())

    class PredictTab(QWidget):
        BATCH_DISPLAY_FORMATS = {
            "R_avg (Å)": "{:.4f}", "Φ_TM (Å⁻¹)": "{:.4f}",
            "Φ_cation (nm⁻¹)": "{:.4f}", "Φ_Na^phase (nm⁻¹)": "{:.4f}",
            "Boundary Φ_Na^phase (nm⁻¹)": "{:.4f}",
            "Phase margin m (nm⁻¹)": "{:+.4f}",
            "Effective ΔX_Na": "{:.2f}", "Predicted ΔV (%)": "{:+.2f}",
        }

        def __init__(self, parent=None):
            super().__init__(parent)
            self.element_inputs = {}
            self.batch_df = None
            self.comparison_items = []
            root = QHBoxLayout(self)
            self.stack = QStackedWidget()
            root.addWidget(self.stack, 2)

            self.page_select = QWidget()
            select_layout = QVBoxLayout(self.page_select)
            select_layout.addStretch()
            title = QLabel("Select Prediction Mode")
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 20px;")
            select_layout.addWidget(title)
            self.btn_go_formula = QPushButton("Formula Input")
            self.btn_go_formula.setFixedHeight(50)
            self.btn_go_formula.setStyleSheet("font-size: 14px;")
            self.btn_go_formula.clicked.connect(lambda: self.stack.setCurrentIndex(1))
            select_layout.addWidget(self.btn_go_formula)
            select_layout.addSpacing(20)
            self.btn_go_designer = QPushButton("Element Designer")
            self.btn_go_designer.setFixedHeight(50)
            self.btn_go_designer.setStyleSheet("font-size: 14px;")
            self.btn_go_designer.clicked.connect(lambda: self.stack.setCurrentIndex(2))
            select_layout.addWidget(self.btn_go_designer)
            select_layout.addStretch()
            self.stack.addWidget(self.page_select)

            self.page_formula = QWidget()
            formula_root = QVBoxLayout(self.page_formula)
            back_formula = QPushButton("← Back")
            back_formula.clicked.connect(lambda: self.stack.setCurrentIndex(0))
            formula_root.addWidget(back_formula)

            formula_box = QGroupBox("Formula Input")
            formula_form = QFormLayout(formula_box)
            self.formula_edit = QLineEdit("Na0.92Ni0.4Mn0.4Al0.05Ti0.1Li0.03Sb0.02O2")
            self.formula_edit.textChanged.connect(self._on_formula_text_changed)
            formula_form.addRow("Formula", self.formula_edit)
            self.formula_charge_lbl = QLabel("--")
            self.formula_tmsum_lbl = QLabel("--")
            self.formula_phase_lbl = QLabel("--")
            formula_form.addRow("Charge (+)", self.formula_charge_lbl)
            formula_form.addRow(rich_label(MATH_HTML["TMSum"]), self.formula_tmsum_lbl)
            formula_form.addRow("Phase gate", self.formula_phase_lbl)
            self.formula_dx = QDoubleSpinBox()
            self.formula_dx.setRange(0.0, 2.0)
            self.formula_dx.setSingleStep(0.01)
            self.formula_dx.setDecimals(2)
            self.formula_dx.setValue(0.60)
            self.formula_qch = QDoubleSpinBox()
            self.formula_qch.setRange(0.0, 500.0)
            self.formula_qch.setSingleStep(0.1)
            self.formula_qch.setDecimals(1)
            self.formula_qch.setValue(149.2)
            self.formula_mode_dx, formula_mode_dx_widget = rich_radio(
                "Direct input", MATH_HTML["DeltaXNa"]
            )
            self.formula_mode_qch, formula_mode_qch_widget = rich_radio(
                "Input charge capacity", MATH_HTML["Qch"], " (mAh g⁻¹)"
            )
            self.formula_mode_dx.setChecked(True)
            formula_mode_group = QButtonGroup(self)
            formula_mode_group.addButton(self.formula_mode_dx)
            formula_mode_group.addButton(self.formula_mode_qch)
            self.formula_dx.valueChanged.connect(self._sync_formula_from_dx)
            self.formula_qch.valueChanged.connect(self._sync_formula_from_qch)
            self.formula_mode_dx.toggled.connect(self._refresh_formula_na_guard)
            self.formula_mode_qch.toggled.connect(self._refresh_formula_na_guard)
            formula_form.addRow(formula_mode_dx_widget)
            formula_form.addRow(rich_label(MATH_HTML["DeltaXNa"]), self.formula_dx)
            formula_form.addRow(formula_mode_qch_widget)
            formula_form.addRow(
                rich_label(MATH_HTML["Qch"], " (mAh g⁻¹)"), self.formula_qch
            )
            self.formula_link_hint = QLabel(
                "Linked inputs: changing the Na-extraction amount or charge capacity "
                "automatically updates the other value."
            )
            self.formula_link_hint.setStyleSheet("color: #666D75; font-size: 11px;")
            formula_form.addRow(self.formula_link_hint)
            self.btn_formula = QPushButton(PREDICT_BUTTON_LABEL)
            self.btn_formula.clicked.connect(self.run_formula)
            formula_form.addRow(self.btn_formula)
            formula_root.addWidget(formula_box)

            batch_box = QGroupBox("Batch Prediction")
            batch_layout = QVBoxLayout(batch_box)
            batch_hint = QLabel(
                "Input file requires 'Formula' plus the manuscript column '"
                f"{MATH_HTML['DeltaXNa']}' or '{MATH_HTML['Qch']} (mAh g⁻¹)'."
            )
            batch_hint.setWordWrap(True)
            batch_hint.setStyleSheet("color: gray; font-size: 11px;")
            batch_layout.addWidget(batch_hint)
            batch_top = QHBoxLayout()
            self.batch_lbl = QLabel("No file loaded")
            self.btn_batch_load = QPushButton("Load CSV/Excel")
            self.btn_batch_load.clicked.connect(self.batch_load_file)
            self.btn_batch_export = QPushButton("Export Results")
            self.btn_batch_export.clicked.connect(self.batch_export_file)
            batch_top.addWidget(self.batch_lbl)
            batch_top.addWidget(self.btn_batch_load)
            batch_top.addWidget(self.btn_batch_export)
            batch_layout.addLayout(batch_top)
            self.batch_table = QTableWidget()
            self.batch_header = RichTextHeaderView(
                Qt.Orientation.Horizontal, self.batch_table
            )
            self.batch_table.setHorizontalHeader(self.batch_header)
            batch_layout.addWidget(self.batch_table)
            formula_root.addWidget(batch_box)

            formula_scroll = QScrollArea()
            formula_scroll.setWidgetResizable(True)
            formula_scroll.setWidget(self.page_formula)
            formula_wrapper = QWidget()
            formula_wrapper_layout = QVBoxLayout(formula_wrapper)
            formula_wrapper_layout.setContentsMargins(0, 0, 0, 0)
            formula_wrapper_layout.addWidget(formula_scroll)
            self.stack.addWidget(formula_wrapper)

            self.page_designer = QWidget()
            designer_root = QVBoxLayout(self.page_designer)
            back_designer = QPushButton("← Back")
            back_designer.clicked.connect(lambda: self.stack.setCurrentIndex(0))
            designer_root.addWidget(back_designer)
            designer_box = QGroupBox("Element Designer")
            designer_layout = QVBoxLayout(designer_box)
            allowed = [symbol for symbol, element in ELEMENTS.items() if element.is_tm]
            self.selector = PeriodicSelector(allowed)
            self.selector.selectionChanged.connect(self.refresh_element_inputs)
            designer_layout.addWidget(self.selector)
            base_form = QFormLayout()
            self.na_spin = QDoubleSpinBox()
            self.na_spin.setRange(0.0, 2.0)
            self.na_spin.setSingleStep(0.01)
            self.na_spin.setValue(0.92)
            self.o_spin = QDoubleSpinBox()
            self.o_spin.setRange(0.0, 4.0)
            self.o_spin.setSingleStep(0.1)
            self.o_spin.setValue(2.0)
            self.na_spin.valueChanged.connect(self._on_designer_composition_changed)
            self.o_spin.valueChanged.connect(self._on_designer_composition_changed)
            base_form.addRow(rich_label(MATH_HTML["XNa"]), self.na_spin)
            base_form.addRow(rich_label(MATH_HTML["nO"]), self.o_spin)
            designer_layout.addLayout(base_form)
            self.element_form = QFormLayout()
            designer_layout.addLayout(self.element_form)
            metrics_form = QFormLayout()
            self.designer_charge_lbl = QLabel("--")
            self.designer_tmsum_lbl = QLabel("--")
            self.designer_phase_lbl = QLabel("--")
            metrics_form.addRow("Charge (+)", self.designer_charge_lbl)
            metrics_form.addRow(rich_label(MATH_HTML["TMSum"]), self.designer_tmsum_lbl)
            metrics_form.addRow("Phase gate", self.designer_phase_lbl)
            designer_layout.addLayout(metrics_form)
            self.refresh_element_inputs(self.selector.selected_elements())
            self.des_dx = QDoubleSpinBox()
            self.des_dx.setRange(0.0, 2.0)
            self.des_dx.setSingleStep(0.01)
            self.des_dx.setDecimals(2)
            self.des_dx.setValue(0.65)
            self.des_qch = QDoubleSpinBox()
            self.des_qch.setRange(0.0, 500.0)
            self.des_qch.setSingleStep(0.1)
            self.des_qch.setDecimals(1)
            self.des_qch.setValue(155.8)
            self.des_mode_dx, des_mode_dx_widget = rich_radio(
                "Direct input", MATH_HTML["DeltaXNa"]
            )
            self.des_mode_qch, des_mode_qch_widget = rich_radio(
                "Input charge capacity", MATH_HTML["Qch"], " (mAh g⁻¹)"
            )
            self.des_mode_dx.setChecked(True)
            designer_mode_group = QButtonGroup(self)
            designer_mode_group.addButton(self.des_mode_dx)
            designer_mode_group.addButton(self.des_mode_qch)
            self.des_dx.valueChanged.connect(self._sync_designer_from_dx)
            self.des_qch.valueChanged.connect(self._sync_designer_from_qch)
            self.des_mode_dx.toggled.connect(self._refresh_designer_na_guard)
            self.des_mode_qch.toggled.connect(self._refresh_designer_na_guard)
            designer_layout.addWidget(des_mode_dx_widget)
            designer_layout.addWidget(rich_label(MATH_HTML["DeltaXNa"]))
            designer_layout.addWidget(self.des_dx)
            designer_layout.addWidget(des_mode_qch_widget)
            designer_layout.addWidget(rich_label(MATH_HTML["Qch"], " (mAh g⁻¹)"))
            designer_layout.addWidget(self.des_qch)
            self.designer_link_hint = QLabel(
                "Linked inputs: changing the Na-extraction amount or charge capacity "
                "automatically updates the other value."
            )
            self.designer_link_hint.setStyleSheet("color: #666D75; font-size: 11px;")
            designer_layout.addWidget(self.designer_link_hint)
            self.btn_designer = QPushButton(PREDICT_BUTTON_LABEL)
            self.btn_designer.clicked.connect(self.run_designer)
            designer_layout.addWidget(self.btn_designer)
            designer_scroll = QScrollArea()
            designer_scroll.setWidgetResizable(True)
            designer_scroll.setWidget(designer_box)
            designer_root.addWidget(designer_scroll)
            self.stack.addWidget(self.page_designer)

            self.update_formula_metrics()
            self.update_designer_metrics()

            results_panel = QWidget()
            results_layout = QVBoxLayout(results_panel)
            results_layout.setContentsMargins(0, 0, 0, 0)
            self.output = QTextEdit()
            self.output.setReadOnly(True)
            self.output.setMaximumHeight(190)
            results_layout.addWidget(self.output)

            comparison_box = QGroupBox("ΔV Comparison")
            comparison_layout = QVBoxLayout(comparison_box)
            comparison_toolbar = QHBoxLayout()
            comparison_hint = QLabel(
                "Each successful prediction is added to the comparison. "
                "Batch input replaces the current comparison set."
            )
            comparison_hint.setStyleSheet("color: #666D75; font-size: 11px;")
            self.btn_clear_comparison = QPushButton("Clear comparison")
            self.btn_clear_comparison.clicked.connect(self._clear_comparison)
            comparison_toolbar.addWidget(comparison_hint, 1)
            comparison_toolbar.addWidget(self.btn_clear_comparison)
            comparison_layout.addLayout(comparison_toolbar)

            comparison_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.comparison_table = QTableWidget(0, 7)
            comparison_headers = [
                "No.", "Candidate", "Formula", "R_avg (Å)",
                "Φ_TM (Å⁻¹)", "ΔX_Na", "Predicted ΔV (%)",
            ]
            self.comparison_table.setHorizontalHeaderLabels(comparison_headers)
            self.comparison_header = RichTextHeaderView(
                Qt.Orientation.Horizontal, self.comparison_table
            )
            self.comparison_table.setHorizontalHeader(self.comparison_header)
            self.comparison_header.set_rich_labels(
                [rich_column_header(column) for column in comparison_headers]
            )
            self.comparison_table.setAlternatingRowColors(True)
            self.comparison_table.verticalHeader().setVisible(False)
            self.comparison_table.setMinimumWidth(350)
            comparison_splitter.addWidget(self.comparison_table)

            self.comparison_chart = DeltaVComparisonChart()
            chart_scroll = QScrollArea()
            chart_scroll.setWidgetResizable(True)
            chart_scroll.setWidget(self.comparison_chart)
            comparison_splitter.addWidget(chart_scroll)
            comparison_splitter.setSizes([440, 560])
            comparison_layout.addWidget(comparison_splitter, 1)
            results_layout.addWidget(comparison_box, 1)
            root.addWidget(results_panel, 3)

        @staticmethod
        def _comparison_record(candidate, result):
            return {
                "candidate": str(candidate),
                "formula": str(result["formula"]),
                "Ravg": float(result["Ravg"]),
                "PhiTM": float(result["PhiTM"]),
                "DeltaXNa": float(result["DeltaXNa"]),
                "DeltaV_pred": float(result["DeltaV_pred"]),
                "DeltaV_lower": float(result["DeltaV_lower"]),
                "DeltaV_upper": float(result["DeltaV_upper"]),
            }

        def _append_comparison_result(self, title, result):
            next_number = len(self.comparison_items) + 1
            candidate = title if title not in {"Formula", "Designer"} else f"C{next_number}"
            self.comparison_items.append(self._comparison_record(candidate, result))
            if len(self.comparison_items) > 30:
                self.comparison_items = self.comparison_items[-30:]
            self._refresh_comparison()

        def _set_comparison_results(self, entries):
            self.comparison_items = [
                self._comparison_record(candidate or f"C{index}", result)
                for index, (candidate, result) in enumerate(entries, start=1)
            ][:50]
            self._refresh_comparison()

        def _clear_comparison(self):
            self.comparison_items = []
            self.comparison_table.setRowCount(0)
            self.comparison_chart.set_items([])

        def _refresh_comparison(self):
            self.comparison_table.setRowCount(len(self.comparison_items))
            for row, item in enumerate(self.comparison_items):
                values = [
                    str(row + 1),
                    item["candidate"],
                    item["formula"],
                    f"{item['Ravg']:.4f}",
                    f"{item['PhiTM']:.4f}",
                    f"{item['DeltaXNa']:.2f}",
                    f"{item['DeltaV_pred']:+.2f}",
                ]
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    if column == 2:
                        cell.setToolTip(item["formula"])
                    elif column != 1:
                        cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.comparison_table.setItem(row, column, cell)
            self.comparison_table.resizeColumnsToContents()
            self.comparison_table.setColumnWidth(2, min(300, self.comparison_table.columnWidth(2)))
            self.comparison_chart.set_items(self.comparison_items)

        def refresh_element_inputs(self, selected):
            while self.element_form.rowCount() > 0:
                self.element_form.removeRow(0)
            self.element_inputs = {}
            for symbol in selected:
                spin = QDoubleSpinBox()
                spin.setRange(0.0, 2.0)
                spin.setSingleStep(0.01)
                spin.setValue(0.0)
                spin.valueChanged.connect(self._on_designer_composition_changed)
                self.element_inputs[symbol] = spin
                element = ELEMENTS[symbol]
                self.element_form.addRow(
                    f"{symbol}  (+{element.valence:g}, {element.radius:g} Å)", spin
                )
            self._on_designer_composition_changed()

        @staticmethod
        def _set_metric_label(label, value, target):
            label.setText(f"{value:.4f}")
            if abs(value - target) < 1e-6:
                label.setStyleSheet("color: green; font-weight: bold;")
            else:
                label.setStyleSheet("color: red; font-weight: bold;")

        @staticmethod
        def _set_phase_label(label, phase):
            margin = phase["phase_margin"]
            suffix = "" if margin is None else f" (m={margin:+.4f} nm⁻¹)"
            label.setText(f"{phase['phase_label']}{suffix}")
            color = {
                "O3": "#2E7D32",
                "P2": "#B3261E",
                "BOUNDARY": "#A15C00",
                "NOT_ASSESSABLE": "#A15C00",
            }[phase["phase_code"]]
            label.setStyleSheet(f"color: {color}; font-weight: bold;")
            label.setToolTip(
                "Zhao cationic-potential composition criterion. "
                "This does not guarantee synthesis of a single O3 phase."
            )

        def update_formula_metrics(self):
            text = self.formula_edit.text().strip()
            if not text:
                for label in (
                    self.formula_charge_lbl,
                    self.formula_tmsum_lbl,
                    self.formula_phase_lbl,
                ):
                    label.setText("--")
                    label.setStyleSheet("")
                return
            try:
                composition = parse_formula(text)
                descriptors = calculate_descriptors(composition)
                self._set_metric_label(self.formula_charge_lbl, positive_charge(composition), 4.0)
                self._set_metric_label(self.formula_tmsum_lbl, descriptors["TM_sum"], 1.0)
                self._set_phase_label(
                    self.formula_phase_lbl,
                    assess_stacking_phase(composition, descriptors),
                )
            except Exception:
                for label in (
                    self.formula_charge_lbl,
                    self.formula_tmsum_lbl,
                    self.formula_phase_lbl,
                ):
                    label.setText("Invalid")
                    label.setStyleSheet("color: red; font-weight: bold;")
            self._refresh_formula_na_guard()

        def _refresh_formula_na_guard(self, *_args):
            if not all(
                hasattr(self, name)
                for name in (
                    "formula_edit", "formula_dx", "formula_qch",
                    "formula_mode_dx", "formula_link_hint", "btn_formula",
                )
            ):
                return
            try:
                composition = parse_formula(self.formula_edit.text().strip())
                descriptors = calculate_descriptors(composition)
                raw_delta_x_na = self.formula_dx.value()
                if not self.formula_mode_dx.isChecked():
                    raw_delta_x_na = (
                        self.formula_qch.value()
                        * descriptors["molar_mass"]
                        * 3.6
                        / FARADAY
                    )
                effective = validate_delta_x_na(raw_delta_x_na, descriptors["XNa"])
                self.btn_formula.setEnabled(True)
                self.formula_link_hint.setText(
                    "Linked inputs. Allowed range: 0 ≤ "
                    f"{MATH_HTML['DeltaXNa']} ≤ {MATH_HTML['XNa']} "
                    f"({descriptors['XNa']:.6g}); model input "
                    f"{MATH_HTML['DeltaXNa']} = {effective:.2f}."
                )
                self.formula_link_hint.setStyleSheet(
                    "color: #666D75; font-size: 11px;"
                )
            except Exception as exc:
                self.btn_formula.setEnabled(False)
                self.formula_link_hint.setText(
                    f"Blocked: {manuscript_html_text(exc)}"
                )
                self.formula_link_hint.setStyleSheet(
                    "color: #B3261E; font-size: 11px; font-weight: bold;"
                )

        @staticmethod
        def _set_linked_value(widget, value):
            blocker = QSignalBlocker(widget)
            widget.setValue(value)
            del blocker

        def _formula_molar_mass(self):
            composition = parse_formula(self.formula_edit.text().strip())
            return calculate_descriptors(composition)["molar_mass"]

        def _sync_formula_from_dx(self, *_args):
            try:
                mass = self._formula_molar_mass()
                qch = self.formula_dx.value() * FARADAY / (mass * 3.6)
                self._set_linked_value(
                    self.formula_qch, round_half_up(qch, QCH_DISPLAY_DECIMALS)
                )
            except Exception:
                pass
            self._refresh_formula_na_guard()

        def _sync_formula_from_qch(self, *_args):
            try:
                mass = self._formula_molar_mass()
                delta_x_na = self.formula_qch.value() * mass * 3.6 / FARADAY
                self._set_linked_value(
                    self.formula_dx, canonical_delta_x_na(delta_x_na)
                )
            except Exception:
                pass
            self._refresh_formula_na_guard()

        def _on_formula_text_changed(self, *_args):
            self.update_formula_metrics()
            if not hasattr(self, "formula_dx"):
                return
            if self.formula_mode_qch.isChecked():
                self._sync_formula_from_qch()
            else:
                self._sync_formula_from_dx()

        def _designer_composition(self):
            composition = {"Na": self.na_spin.value(), "O": self.o_spin.value()}
            for symbol, spin in self.element_inputs.items():
                if spin.value() > 0:
                    composition[symbol] = spin.value()
            return composition

        def update_designer_metrics(self):
            try:
                composition = self._designer_composition()
                descriptors = calculate_descriptors(composition)
                self._set_metric_label(self.designer_charge_lbl, positive_charge(composition), 4.0)
                self._set_metric_label(self.designer_tmsum_lbl, descriptors["TM_sum"], 1.0)
                self._set_phase_label(
                    self.designer_phase_lbl,
                    assess_stacking_phase(composition, descriptors),
                )
            except Exception:
                for label in (
                    self.designer_charge_lbl,
                    self.designer_tmsum_lbl,
                    self.designer_phase_lbl,
                ):
                    label.setText("Invalid")
                    label.setStyleSheet("color: red; font-weight: bold;")
            self._refresh_designer_na_guard()

        def _refresh_designer_na_guard(self, *_args):
            if not all(
                hasattr(self, name)
                for name in (
                    "des_dx", "des_qch", "des_mode_dx",
                    "designer_link_hint", "btn_designer",
                )
            ):
                return
            try:
                composition = self._designer_composition()
                descriptors = calculate_descriptors(composition)
                raw_delta_x_na = self.des_dx.value()
                if not self.des_mode_dx.isChecked():
                    raw_delta_x_na = (
                        self.des_qch.value()
                        * descriptors["molar_mass"]
                        * 3.6
                        / FARADAY
                    )
                effective = validate_delta_x_na(raw_delta_x_na, descriptors["XNa"])
                self.btn_designer.setEnabled(True)
                self.designer_link_hint.setText(
                    "Linked inputs. Allowed range: 0 ≤ "
                    f"{MATH_HTML['DeltaXNa']} ≤ {MATH_HTML['XNa']} "
                    f"({descriptors['XNa']:.6g}); model input "
                    f"{MATH_HTML['DeltaXNa']} = {effective:.2f}."
                )
                self.designer_link_hint.setStyleSheet(
                    "color: #666D75; font-size: 11px;"
                )
            except Exception as exc:
                self.btn_designer.setEnabled(False)
                self.designer_link_hint.setText(
                    f"Blocked: {manuscript_html_text(exc)}"
                )
                self.designer_link_hint.setStyleSheet(
                    "color: #B3261E; font-size: 11px; font-weight: bold;"
                )

        def _designer_molar_mass(self):
            return calculate_descriptors(self._designer_composition())["molar_mass"]

        def _sync_designer_from_dx(self, *_args):
            try:
                mass = self._designer_molar_mass()
                qch = self.des_dx.value() * FARADAY / (mass * 3.6)
                self._set_linked_value(
                    self.des_qch, round_half_up(qch, QCH_DISPLAY_DECIMALS)
                )
            except Exception:
                pass
            self._refresh_designer_na_guard()

        def _sync_designer_from_qch(self, *_args):
            try:
                mass = self._designer_molar_mass()
                delta_x_na = self.des_qch.value() * mass * 3.6 / FARADAY
                self._set_linked_value(self.des_dx, canonical_delta_x_na(delta_x_na))
            except Exception:
                pass
            self._refresh_designer_na_guard()

        def _on_designer_composition_changed(self, *_args):
            self.update_designer_metrics()
            if not hasattr(self, "des_dx"):
                return
            if self.des_mode_qch.isChecked():
                self._sync_designer_from_qch()
            else:
                self._sync_designer_from_dx()

        def _render_result(self, title, result):
            if result["prediction_allowed"]:
                status = "WARN" if result["warnings"] else "PASS"
            else:
                status = "BLOCKED" if result["phase_code"] == "P2" else "REVIEW"
            lines = [
                f"<b>[{html.escape(str(title))}]</b>",
                f"<b>Status:</b> {status}",
                f"<b>Zhao phase criterion:</b> {html.escape(result['phase_label'])}",
                f"{MATH_HTML['XNa']} = {result['XNa']:.4f}; "
                f"{MATH_HTML['TMSum']} = {result['TM_sum']:.4f}; "
                f"{MATH_HTML['Ravg']} = {result['Ravg']:.4f} Å; "
                f"{MATH_HTML['PhiTM']} = {result['PhiTM']:.4f} Å<sup>−1</sup>",
                f"Charge (+): {positive_charge(parse_formula(result['formula'])):.4f}",
                f"Model input {MATH_HTML['DeltaXNa']} = {result['DeltaXNa']:.2f}",
            ]
            if result["PhiCation_phase_nm_inv"] is not None:
                lines.extend(
                    [
                        f"Phase-map coordinates: {MATH_HTML['PhiCation']} = "
                        f"{result['PhiCation_phase_nm_inv']:.4f} nm⁻¹, "
                        f"{MATH_HTML['PhiNaPhase']} = "
                        f"{result['PhiNa_phase_nm_inv']:.4f} nm⁻¹",
                        f"Boundary {MATH_HTML['PhiNaPhase']} = "
                        f"{result['phase_boundary_phi_na']:.4f} nm⁻¹, "
                        f"phase margin <i>m</i> = {result['phase_margin']:+.4f} nm⁻¹",
                    ]
                )
            lines.append(
                "Scope note: the composition rule screens P2/O3 stacking preference; "
                "it does not guarantee synthesis of a single O3 phase."
            )
            if result["prediction_allowed"]:
                lines.extend(
                    [
                        f"Risk: {result['risk']}",
                        f"Predicted {MATH_HTML['DeltaV']}: "
                        f"{result['DeltaV_pred']:+.2f}%",
                        "CV-based empirical reference interval: "
                        f"[{result['DeltaV_lower']:.2f}%, {result['DeltaV_upper']:.2f}%]",
                        "Target note: signed ΔV; negative values indicate volume contraction. "
                        "This interval is not an independently validated 95% coverage claim.",
                    ]
                )
            else:
                lines.append(
                    "Predicted ΔV: BLOCKED - only O3-favored candidates pass the phase gate."
                )
            if result["phase_gate_reasons"]:
                lines.append("<b>Phase-gate reasons:</b>")
                lines.extend(
                    manuscript_html_text(f"• {reason}")
                    for reason in result["phase_gate_reasons"]
                )
            if result["warnings"]:
                lines.append("<b>Warnings:</b>")
                lines.extend(
                    manuscript_html_text(f"• {warning}")
                    for warning in result["warnings"]
                )
            self.output.append("<div style='margin-bottom:10px'>" + "<br>".join(lines) + "</div>")
            if result["prediction_allowed"]:
                self._append_comparison_result(title, result)

        def run_formula(self):
            try:
                kwargs = (
                    {"delta_x_na": self.formula_dx.value()}
                    if self.formula_mode_dx.isChecked()
                    else {"qch": self.formula_qch.value()}
                )
                self._render_result(
                    "Formula", predict_formula(self.formula_edit.text().strip(), **kwargs)
                )
            except Exception as exc:
                message = manuscript_html_text(exc)
                self.output.append(f"<b>[Formula] ERROR:</b> {message}<br>")
                QMessageBox.critical(self, "Prediction Error", message)

        def run_designer(self):
            try:
                composition = self._designer_composition()
                if self.des_mode_dx.isChecked():
                    delta_x_na = self.des_dx.value()
                else:
                    mass = calculate_descriptors(composition)["molar_mass"]
                    delta_x_na = self.des_qch.value() * mass * 3.6 / FARADAY
                self._render_result(
                    "Designer", predict_from_composition(composition, delta_x_na)
                )
            except Exception as exc:
                message = manuscript_html_text(exc)
                self.output.append(f"<b>[Designer] ERROR:</b> {message}<br>")
                QMessageBox.critical(self, "Prediction Error", message)

        def batch_load_file(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select File", "", "Data Files (*.csv *.xlsx)"
            )
            if not path:
                return
            try:
                source = Path(path)
                frame = (
                    pd.read_excel(source)
                    if source.suffix.lower() == ".xlsx"
                    else pd.read_csv(source)
                )
                formula_col = next(
                    (column for column in frame.columns if str(column).strip().lower() == "formula"),
                    None,
                )
                delta_col = next(
                    (
                        column for column in frame.columns
                        if normalize_input_column_name(column) in {"deltaxna", "δxna"}
                    ),
                    None,
                )
                qch_col = next(
                    (
                        column for column in frame.columns
                        if normalize_input_column_name(column).startswith("qch")
                    ),
                    None,
                )
                name_col = next(
                    (
                        column for column in frame.columns
                        if str(column).strip().lower()
                        in {
                            "name", "sample", "candidate", "remark", "remarks",
                        }
                    ),
                    None,
                )
                if formula_col is None or (delta_col is None and qch_col is None):
                    raise ValueError(
                        "Input file must contain Formula and ΔX_Na or Q_ch. "
                        "Legacy DeltaXNa/Q-ch headers are also accepted."
                    )
                rows = []
                comparison_entries = []
                for row_number, (_, source_row) in enumerate(frame.iterrows(), start=1):
                    try:
                        kwargs = (
                            {"delta_x_na": float(source_row[delta_col])}
                            if delta_col is not None and pd.notna(source_row[delta_col])
                            else {"qch": float(source_row[qch_col])}
                        )
                        result = predict_formula(str(source_row[formula_col]), **kwargs)
                        candidate = (
                            str(source_row[name_col]).strip()
                            if name_col is not None and pd.notna(source_row[name_col])
                            else f"C{row_number}"
                        )
                        if result["prediction_allowed"]:
                            comparison_entries.append((candidate, result))
                        if result["prediction_allowed"]:
                            status = "WARN" if result["warnings"] else "PASS"
                        else:
                            status = (
                                "BLOCKED" if result["phase_code"] == "P2" else "REVIEW"
                            )
                        messages = list(result["phase_gate_reasons"]) + list(
                            result["warnings"]
                        )
                        rows.append(
                            {
                                "Formula": result["formula"],
                                "R_avg (Å)": result["Ravg"],
                                "Φ_TM (Å⁻¹)": result["PhiTM"],
                                "Phase": result["phase_label"],
                                "Φ_cation (nm⁻¹)": result["PhiCation_phase_nm_inv"],
                                "Φ_Na^phase (nm⁻¹)": result["PhiNa_phase_nm_inv"],
                                "Boundary Φ_Na^phase (nm⁻¹)": result[
                                    "phase_boundary_phi_na"
                                ],
                                "Phase margin m (nm⁻¹)": result["phase_margin"],
                                "Prediction allowed": (
                                    "Yes" if result["prediction_allowed"] else "No"
                                ),
                                "Effective ΔX_Na": result["DeltaXNa"],
                                "Predicted ΔV (%)": result["DeltaV_pred"],
                                "Status": status,
                                "Risk level": result["risk"] or "",
                                "Warnings": "; ".join(messages),
                            }
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "Formula": str(source_row.get(formula_col, "")),
                                "Prediction allowed": "No",
                                "Predicted ΔV (%)": None,
                                "Status": "FAIL",
                                "Risk level": "",
                                "Warnings": str(exc),
                            }
                        )
                self.batch_df = pd.DataFrame(rows)
                self.batch_lbl.setText(path)
                self._batch_render_table()
                self._set_comparison_results(comparison_entries)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Batch Error", manuscript_html_text(exc)
                )

        def _batch_format_cell(self, column, value):
            if pd.isna(value):
                return ""
            value_format = self.BATCH_DISPLAY_FORMATS.get(column)
            if value_format and isinstance(value, (int, float, np.integer, np.floating)):
                return value_format.format(float(value))
            return str(value)

        def _batch_render_table(self):
            if self.batch_df is None:
                return
            frame = self.batch_df
            self.batch_table.clear()
            self.batch_table.setRowCount(len(frame))
            self.batch_table.setColumnCount(len(frame.columns))
            self.batch_table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
            self.batch_header.set_rich_labels(
                [rich_column_header(str(column)) for column in frame.columns]
            )
            for row in range(len(frame)):
                for column, name in enumerate(frame.columns):
                    self.batch_table.setItem(
                        row, column,
                        QTableWidgetItem(self._batch_format_cell(name, frame.iloc[row, column])),
                    )
            self.batch_table.resizeColumnsToContents()

        def batch_export_file(self):
            if self.batch_df is None:
                QMessageBox.information(self, "Info", "Please load and run batch prediction first")
                return
            path, selected_filter = QFileDialog.getSaveFileName(
                self, "Save Results", "batch_predictions.csv", "CSV (*.csv);;Excel (*.xlsx)"
            )
            if not path:
                return
            try:
                export_frame = self.batch_df.copy()
                for column, decimals in {
                    "R_avg (Å)": 4,
                    "Φ_TM (Å⁻¹)": 4,
                    "Φ_cation (nm⁻¹)": 4,
                    "Φ_Na^phase (nm⁻¹)": 4,
                    "Boundary Φ_Na^phase (nm⁻¹)": 4,
                    "Phase margin m (nm⁻¹)": 4,
                    "Effective ΔX_Na": 2,
                    "Predicted ΔV (%)": 2,
                }.items():
                    if column in export_frame.columns:
                        export_frame[column] = pd.to_numeric(
                            export_frame[column], errors="coerce"
                        ).round(decimals)
                if path.lower().endswith(".xlsx") or "Excel" in selected_filter:
                    if not path.lower().endswith(".xlsx"):
                        path += ".xlsx"
                    export_frame.to_excel(path, index=False)
                else:
                    if not path.lower().endswith(".csv"):
                        path += ".csv"
                    export_frame.to_csv(path, index=False, encoding="utf-8-sig")
                QMessageBox.information(self, "Done", f"Results saved: {path}")
            except Exception as exc:
                QMessageBox.critical(self, "Export Error", str(exc))

    class ModelInfoTab(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            layout = QVBoxLayout(self)
            self.text = QTextEdit()
            self.text.setReadOnly(True)
            layout.addWidget(self.text)
            self.refresh()

        def refresh(self):
            delta_meta = MODEL_META["DeltaV"]
            fixed = delta_meta["fixed_model_repeated_20x5_cv_metrics"]
            nested = delta_meta["nested_model_selection_10x5_cv_metrics"]
            apparent = delta_meta["apparent_training_metrics"]
            audit = delta_meta["overfitting_audit"]
            hyper = delta_meta["hyperparameters"]
            interval = delta_meta["prediction_interval"]
            domain_rows = []
            for symbol, key, unit in (
                (MATH_HTML["Ravg"], "Ravg", "Å"),
                (MATH_HTML["PhiTM"], "PhiTM", "Å<sup>−1</sup>"),
                (MATH_HTML["DeltaXNa"], "DeltaXNa", "dimensionless"),
            ):
                bounds = TRAINING_DOMAIN[key]
                domain_rows.append(
                    f"<tr><td>{symbol}</td><td>{bounds['min']:.4f}</td>"
                    f"<td>{bounds['max']:.4f}</td><td>{unit}</td></tr>"
                )
            model_information = f"""
            <html><head><style>
              body {{ font-family: 'Arial'; color: #27313a; line-height: 1.35; }}
              h2 {{ color: #3f5f8e; margin-top: 16px; margin-bottom: 6px; }}
              h3 {{ color: #4d5661; margin-top: 12px; margin-bottom: 4px; }}
              table {{ border-collapse: collapse; margin: 5px 0 10px 0; }}
              th, td {{ border: 1px solid #c8ced6; padding: 5px 8px; }}
              th {{ background: #eef2f6; }}
              .note {{ background: #f6f2e9; border: 1px solid #d8c9aa; padding: 7px; }}
            </style></head><body>
            <h2>EM-VRF Desktop</h2>
            <p><b>Application version:</b> {html.escape(APP_VERSION)}<br>
            <b>Model version:</b> {html.escape(str(MODEL_META.get('version', 'not reported')))}</p>

            <h3>Signed volume-response model</h3>
            <p><b>Inputs:</b> {MATH_HTML['Ravg']} (Å), {MATH_HTML['PhiTM']}
            (Å<sup>−1</sup>), and {MATH_HTML['DeltaXNa']}.<br>
            <b>Target:</b> signed {MATH_HTML['DeltaV']} (%); |Δ<i>V</i>| is not
            used as the regression target.<br>
            <b>Estimator:</b> StandardScaler + RBF-SVR
            (<i>C</i> = {hyper['C']:g}, ε = {hyper['epsilon']:g}, γ =
            {html.escape(str(hyper['gamma']))}).<br>
            <b>Training population:</b> {delta_meta['unified_literature_rows']} fixed
            literature records.</p>

            <table>
              <tr><th>Evaluation</th><th>R²</th><th>RMSE (%)</th><th>MAE (%)</th><th>Pearson r</th><th>n</th></tr>
              <tr><td>Apparent training fit</td><td>{apparent['R2']:.3f}</td><td>{apparent['RMSE']:.3f}</td><td>{apparent['MAE']:.3f}</td><td>{apparent['Pearson_r']:.3f}</td><td>{apparent['n']}</td></tr>
              <tr><td>Aggregated mean-OOF prediction, repeated 20 × 5 CV</td><td>{fixed['R2']:.3f}</td><td>{fixed['RMSE']:.3f}</td><td>{fixed['MAE']:.3f}</td><td>{fixed['Pearson_r']:.3f}</td><td>{fixed['n']}</td></tr>
              <tr><td>Aggregated mean-OOF prediction, nested 10 × 5 CV</td><td>{nested['R2']:.3f}</td><td>{nested['RMSE']:.3f}</td><td>{nested['MAE']:.3f}</td><td>{nested['Pearson_r']:.3f}</td><td>{nested['n']}</td></tr>
            </table>
            <p>The reported repeated-CV metrics are calculated once after averaging each
            record's repeated out-of-fold predictions; they are not the arithmetic mean
            of 20 independently reported fold-repeat metrics.</p>
            <p><b>Overfitting audit:</b> apparent minus repeated-CV R² =
            {audit['apparent_minus_fixed_cv_R2']:.3f}; repeated-CV/apparent RMSE
            ratio = {audit['fixed_cv_to_apparent_RMSE_ratio']:.3f}.</p>

            <h3>Training-domain ranges for the three regression inputs</h3>
            <table><tr><th>Descriptor</th><th>Minimum</th><th>Maximum</th><th>Unit</th></tr>
            {''.join(domain_rows)}</table>
            <p>Additional composition-dispersion checks are diagnostic warnings only;
            they are not regression inputs.</p>

            <h3>Post-processing and reference interval</h3>
            <p>Output constraint:
            <i>ΔV̂</i><sub>output</sub> = min(<i>ΔV̂</i><sub>raw</sub>,
            −1 × 10<sup>−6</sup>%). Thus, the current software does not report a
            predicted positive volume expansion.</p>
            <p>Reference half-width <i>h</i> = {interval['half_width']:.3f} percentage
            points, derived from repeated-CV absolute residuals (calibration n =
            {interval['calibration_n']}). The displayed interval is a CV-based empirical
            reference interval and has not been independently validated to provide 95%
            coverage.</p>

            <h3>P2/O3 phase gate</h3>
            <p>{MATH_HTML['PhiCation']} =
            {MATH_HTML['PhiTMPhase']} × {MATH_HTML['PhiNaPhase']} /
            {MATH_HTML['PhiOPhase']}; all phase-map ionic potentials are in
            nm<sup>−1</sup>.<br>
            Dividing line: {MATH_HTML['PhiNaPhase']} = 0.4065
            {MATH_HTML['PhiCation']} + 1.572.<br>
            Phase margin: <i>m</i> = {MATH_HTML['PhiNaPhase']} −
            (0.4065 {MATH_HTML['PhiCation']} + 1.572).</p>
            <p>O3-favored when <i>m</i> &gt; 10<sup>−9</sup> nm<sup>−1</sup>;
            P2-favored when <i>m</i> &lt; −10<sup>−9</sup> nm<sup>−1</sup>;
            otherwise boundary/uncertain. Applicability requires
            {MATH_HTML['XNa']} &gt; 0, 0.95 ≤ {MATH_HTML['TMSum']} ≤ 1.05,
            |{MATH_HTML['nO']} − 2| ≤ 0.05, and |<i>q</i><sub>gap</sub>| ≤ 0.05.</p>
            <p>Reference: Zhao et al., <i>Science</i> 370, 708–711 (2020),
            DOI: 10.1126/science.aay9972.</p>

            <h3>Input precision and guard</h3>
            <p>{MATH_HTML['DeltaXNa']} is used/displayed to {DELTA_X_MODEL_DECIMALS}
            decimals; {MATH_HTML['Qch']} to {QCH_DISPLAY_DECIMALS} decimal;
            {MATH_HTML['DeltaV']} to {DELTA_V_DISPLAY_DECIMALS} decimals.
            Prediction requires 0 ≤ {MATH_HTML['DeltaXNa']} ≤ {MATH_HTML['XNa']}.</p>

            <div class="note"><b>Scope and interpretation.</b> O3-favored is a
            composition-based stacking preference, not proof of synthesis, phase purity,
            or thermodynamic stability. Cross-validation is not independent external
            validation. Low/Medium/High labels are heuristic |Δ<i>V</i>| bands
            (≤1.0%, 1.0–2.5%, &gt;2.5%), not a trained failure classifier.</div>
            </body></html>
            """
            self.text.setHtml(model_information)

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("EM-VRF Desktop")
            self.resize(1400, 900)
            self.tabs = QTabWidget()
            self.predict_tab = PredictTab()
            self.model_info_tab = ModelInfoTab()
            self.tabs.addTab(self.predict_tab, "Predict")
            self.tabs.addTab(self.model_info_tab, "Model Info")
            self.setCentralWidget(self.tabs)

    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    app_icon = QIcon(str(APP_ICON_PATH))
    app.setWindowIcon(app_icon)
    window = MainWindow()
    window.setWindowIcon(app_icon)
    window.show()
    if smoke_test:
        assert APP_ICON_PATH.is_file()
        assert not app.windowIcon().isNull()
        assert not window.windowIcon().isNull()
        assert window.windowTitle() == "EM-VRF Desktop"
        assert window.width() >= 1200 and window.height() >= 800
        assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
            "Predict", "Model Info"
        ]
        assert window.predict_tab.stack.count() == 3
        assert window.predict_tab.formula_edit.text() == (
            "Na0.92Ni0.4Mn0.4Al0.05Ti0.1Li0.03Sb0.02O2"
        )
        assert abs(window.predict_tab.formula_dx.value() - 0.60) < 1e-12
        assert abs(window.predict_tab.formula_qch.value() - 149.2) < 1e-12
        assert window.predict_tab.formula_dx.decimals() == 2
        assert window.predict_tab.formula_qch.decimals() == 1
        assert window.predict_tab.formula_mode_dx.text() == "Direct input"
        assert window.predict_tab.btn_formula.text() == PREDICT_BUTTON_LABEL
        assert window.predict_tab.btn_designer.text() == PREDICT_BUTTON_LABEL
        assert (
            window.predict_tab.comparison_chart.y_axis_label
            == COMPARISON_Y_AXIS_LABEL
        )
        assert window.predict_tab.comparison_table.horizontalHeaderItem(3).text() == (
            "R_avg (Å)"
        )
        assert window.predict_tab.comparison_table.horizontalHeaderItem(4).text() == (
            "Φ_TM (Å⁻¹)"
        )
        model_info_text = window.model_info_tab.text.toPlainText()
        for expected_text in (
            "Application version", "Signed volume-response model",
            "Aggregated mean-OOF prediction", "Overfitting audit", "P2/O3 phase gate",
            "CV-based empirical reference interval",
        ):
            assert expected_text in model_info_text
        direct_result = predict_formula(
            window.predict_tab.formula_edit.text(), delta_x_na=0.60
        )
        capacity_result = predict_formula(
            window.predict_tab.formula_edit.text(), qch=149.2
        )
        assert direct_result["DeltaXNa"] == capacity_result["DeltaXNa"] == 0.60
        assert direct_result["DeltaV_pred"] == capacity_result["DeltaV_pred"]
        assert "O3-favored" in window.predict_tab.formula_phase_lbl.text()
        formula_mass = calculate_descriptors(
            parse_formula(window.predict_tab.formula_edit.text())
        )["molar_mass"]
        window.predict_tab.formula_dx.setValue(0.61)
        expected_qch = round_half_up(
            0.61 * FARADAY / (formula_mass * 3.6), QCH_DISPLAY_DECIMALS
        )
        assert window.predict_tab.formula_qch.value() == expected_qch
        window.predict_tab.formula_qch.setValue(149.2)
        assert window.predict_tab.formula_dx.value() == 0.60
        window.predict_tab.formula_dx.setValue(0.93)
        assert not window.predict_tab.btn_formula.isEnabled()
        assert "Blocked:" in window.predict_tab.formula_link_hint.text()
        window.predict_tab.formula_dx.setValue(0.60)
        assert window.predict_tab.btn_formula.isEnabled()
        window.predict_tab.formula_mode_qch.setChecked(True)
        window.predict_tab.formula_qch.setValue(229.7)
        assert not window.predict_tab.btn_formula.isEnabled()
        window.predict_tab.formula_qch.setValue(149.2)
        window.predict_tab.formula_mode_dx.setChecked(True)
        assert window.predict_tab.btn_formula.isEnabled()
        blocked_result = predict_formula(
            "Na5/6Li5/18Mn13/18O2", delta_x_na=0.50
        )
        comparison_count = len(window.predict_tab.comparison_items)
        window.predict_tab._render_result("P2 reference", blocked_result)
        assert len(window.predict_tab.comparison_items) == comparison_count
        assert "Predicted ΔV: BLOCKED" in window.predict_tab.output.toPlainText()
        window.predict_tab._clear_comparison()
        window.predict_tab.output.clear()
        if snapshot_path:
            examples = [
                (
                    "NMTS",
                    "Na0.92Ni0.4Mn0.4Al0.05Ti0.1Li0.03Sb0.02O2",
                    149.2,
                ),
                (
                    "NMTS-L",
                    "NaNi0.4Mn0.4Al0.06Ti0.04Li0.06Sb0.04O2",
                    150.2,
                ),
                (
                    "NMTS-A",
                    "Na0.92Ni0.4Mn0.4Al0.08Ti0.08Li0.02Sb0.02O2",
                    139.2,
                ),
                (
                    "NMTS-M",
                    "Na0.905Ni0.4Mn0.4Al0.05Ti0.1Li0.015Mg0.02Mo0.015O2",
                    140.6,
                ),
            ]
            for label, formula, qch in examples:
                window.predict_tab._render_result(
                    label, predict_formula(formula, qch=qch)
                )
            window.predict_tab.stack.setCurrentIndex(1)
            assert window.predict_tab.comparison_table.rowCount() == len(examples)
        else:
            smoke_result = predict_formula(
                window.predict_tab.formula_edit.text(), delta_x_na=0.60
            )
            window.predict_tab._render_result("Formula", smoke_result)
        output_text = window.predict_tab.output.toPlainText()
        assert output_text.count("Predicted ΔV:") == (4 if snapshot_path else 1)
        assert "Predicted ΔV:" in output_text
        for symbol in ("Ru", "Ce"):
            assert symbol in window.predict_tab.selector.checkboxes
            assert window.predict_tab.selector.checkboxes[symbol].isEnabled()
        assert all(
            checkbox.text().strip()
            for checkbox in window.predict_tab.selector.findChildren(QCheckBox)
        )
        app.processEvents()
        if snapshot_path:
            snapshot = window.grab()
            if not snapshot.save(snapshot_path):
                raise RuntimeError(f"Could not save GUI snapshot: {snapshot_path}")
        window.close()
        return 0
    return app.exec()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gui-smoke-test", action="store_true")
    parser.add_argument("--gui-snapshot")
    parser.add_argument("--predict")
    parser.add_argument("--delta-x-na", type=float)
    parser.add_argument("--qch", type=float)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.self_test:
        payload = self_test()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output_json:
            Path(args.output_json).write_text(serialized, encoding="utf-8")
        else:
            print(serialized)
        return 0
    if args.predict:
        payload = public_prediction_payload(
            predict_formula(args.predict, args.delta_x_na, args.qch)
        )
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output_json:
            Path(args.output_json).write_text(serialized, encoding="utf-8")
        else:
            print(serialized)
        return 0
    return run_gui(
        smoke_test=args.gui_smoke_test or bool(args.gui_snapshot),
        snapshot_path=args.gui_snapshot,
    )


# ============================================================================
# Revision 2026-10-03 (r3): full-precision input handling, applied unchanged.
# Same code as precision_revision/full_precision_patch.py (line endings aside).
# ============================================================================
"""EM-VRF 10.3 precision correction. Executed in the original application globals."""
import types as _types

APP_VERSION = "2026.10.03-full-precision-r3"
QCH_DISPLAY_DECIMALS = 2


def canonical_delta_x_na(value):
    """Preserve the floating-point input; formatting belongs to the view."""
    return float(value)


def validate_delta_x_na(delta_x_na, x_na):
    raw_value = float(delta_x_na)
    initial_na = float(x_na)
    if not math.isfinite(raw_value):
        raise ValueError("ΔX_Na must be a finite number.")
    if raw_value < 0:
        raise ValueError(f"ΔX_Na={raw_value:.6g} is invalid. Required range: 0 ≤ ΔX_Na ≤ X_Na ({initial_na:.6g}). Prediction was blocked.")
    if raw_value > initial_na + NA_LIMIT_TOLERANCE:
        raise ValueError(f"ΔX_Na={raw_value:.6g} exceeds the initial Na stoichiometry X_Na={initial_na:.6g}. Required range: 0 ≤ ΔX_Na ≤ X_Na. Prediction was blocked.")
    return raw_value


def _sync_formula_from_dx(self, *args):
    try:
        value = self.formula_dx.value() * FARADAY / (self._formula_molar_mass() * 3.6)
        self._set_linked_value(self.formula_qch, value)
    except Exception:
        pass
    self._refresh_formula_na_guard()


def _sync_designer_from_dx(self, *args):
    try:
        value = self.des_dx.value() * FARADAY / (self._designer_molar_mass() * 3.6)
        self._set_linked_value(self.des_qch, value)
    except Exception:
        pass
    self._refresh_designer_na_guard()


def _export_dataframe_precision(frame, path):
    """XLSX retains numeric values with a two-decimal number format; CSV is raw."""
    if str(path).lower().endswith(".xlsx"):
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False)
            sheet = writer.sheets["Sheet1"]
            for row in sheet.iter_rows(min_row=2):
                for cell in row:
                    if isinstance(cell.value, (float, np.floating)):
                        cell.number_format = "0.00"
    else:
        frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")


def _export_method_factory(QFileDialog, QMessageBox):
    def batch_export_file(self):
        if self.batch_df is None or self.batch_df.empty:
            QMessageBox.information(self, "Export", "No batch results to export.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(self, "Export results", "EM_VRF_predictions.xlsx", "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        if Path(path).suffix.lower() not in (".xlsx", ".csv"):
            path += ".csv" if selected_filter.startswith("CSV") else ".xlsx"
        try:
            _export_dataframe_precision(self.batch_df, path)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
    return batch_export_file


_replacement_codes = {
    "_sync_formula_from_dx": _sync_formula_from_dx.__code__,
    "_sync_designer_from_dx": _sync_designer_from_dx.__code__,
    "batch_export_file": _export_method_factory(None, None).__code__,
}


def _rewrite_nested_precision(code):
    if code.co_name in _replacement_codes:
        replacement = _replacement_codes[code.co_name]
        if replacement.co_freevars != code.co_freevars:
            raise RuntimeError("Precision patch closure mismatch: " + code.co_name)
        return replacement
    constants = []
    for item in code.co_consts:
        if isinstance(item, _types.CodeType):
            item = _rewrite_nested_precision(item)
        elif code.co_name == "refresh" and isinstance(item, str):
            item = item.replace(" is used/displayed to ", " is displayed to ")
            item = item.replace("\n            decimals; ", "\n            decimals (full precision used internally); ")
            item = item.replace(" decimal;\n            ", " decimals;\n            ")
        constants.append(item)
    return code.replace(co_consts=tuple(constants))


_original_run_gui = _types.FunctionType(_rewrite_nested_precision(run_gui.__code__), globals(), "_original_run_gui", run_gui.__defaults__)


def _make_precision_spinbox(base):
    class FullPrecisionSpinBox(base):
        """Full floating-point value, compact idle display, full value when editing."""
        def __init__(self, *args, **kwargs):
            self._editing_precision = False
            self._edited_precision = False
            self._setting_precision = False
            super().__init__(*args, **kwargs)
            super().setDecimals(17)
            self._exact_precision = super().value()
            self.setKeyboardTracking(False)
            self.valueChanged.connect(self._capture_precision)
            self.lineEdit().textEdited.connect(self._mark_precision_edited)

        def setDecimals(self, decimals):
            super().setDecimals(17)

        def _mark_precision_edited(self, text):
            self._edited_precision = True

        def _capture_precision(self, value):
            self._qt_precision_signal = True
            if not self._setting_precision:
                self._exact_precision = float(value)
            self.setToolTip("Full precision: " + repr(self.value()) + "\nClick to edit the full value; the idle display uses two decimals.")

        def value(self):
            return getattr(self, "_exact_precision", super().value())

        def setValue(self, value):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Input must be finite.")
            previous = self.value()
            self._exact_precision = min(max(value, self.minimum()), self.maximum())
            self._qt_precision_signal = False
            self._setting_precision = True
            try:
                super().setValue(self._exact_precision)
            finally:
                self._setting_precision = False
            if previous != self._exact_precision and not self._qt_precision_signal:
                # Qt may map two adjacent floating-point values to the same
                # 17-decimal storage value; linked controls still need a signal.
                self._setting_precision = True
                try:
                    self.valueChanged.emit(self._exact_precision)
                finally:
                    self._setting_precision = False
            self.setToolTip("Full precision: " + repr(self.value()) + "\nClick to edit the full value; the idle display uses two decimals.")

        def textFromValue(self, value):
            if getattr(self, "_editing_precision", False):
                return repr(float(value))
            return format(float(value), ".2f")

        def valueFromText(self, text):
            if not getattr(self, "_edited_precision", False) and hasattr(self, "_exact_precision"):
                return self._exact_precision
            return super().valueFromText(text)

        def focusInEvent(self, event):
            self._editing_precision = True
            self._edited_precision = False
            super().focusInEvent(event)
            self.lineEdit().setText(repr(self.value()))

        def focusOutEvent(self, event):
            super().focusOutEvent(event)
            self._editing_precision = False
            self._edited_precision = False
            self.lineEdit().setText(format(self.value(), ".2f"))

        def stepBy(self, steps):
            self.setValue(self.value() + int(steps) * self.singleStep())
    return FullPrecisionSpinBox


_PRECISION_CASES = [
    ("NMTS", "Na0.92Ni0.4Mn0.4Al0.05Ti0.1Li0.03Sb0.02O2", 149.24, -1.361105620589903),
    ("NMTS-A", "Na0.92Ni0.4Mn0.4Al0.08Ti0.08Li0.02Sb0.02O2", 139.2, -0.4934918803431927),
    ("NMTS-M", "Na0.905Ni0.4Mn0.4Al0.05Ti0.1Li0.015Mg0.02Mo0.015O2", 140.6, -0.6035408535903892),
    ("NMTS-L", "NaNi0.4Mn0.4Al0.06Ti0.04Li0.06Sb0.04O2", 150.2, -2.0087560879279645),
]


def self_test():
    assert ELEMENTS["Ce"].valence == 4 and ELEMENTS["Ce"].radius == 0.87
    ce_formula = "Na0.92Ni0.4Mn0.38Ce0.02Al0.05Ti0.1Li0.03Sb0.02O2"
    ce_descriptors = calculate_descriptors(parse_formula(ce_formula))
    assert abs(ce_descriptors["Ravg"] - 0.61685) < 1e-12
    expected_ce_phi = 5.325959421829658 + 0.02 * (4 / 0.87 - 4 / 0.53)
    assert abs(ce_descriptors["PhiTM"] - expected_ce_phi) < 1e-12
    results = []
    for name, formula, qch, expected in _PRECISION_CASES:
        result = predict_formula(formula, qch=qch)
        mass = calculate_descriptors(parse_formula(formula))["molar_mass"]
        raw_dx = qch * mass * 3.6 / FARADAY
        assert result["DeltaXNa"] == raw_dx
        assert abs(result["DeltaV_pred"] - expected) < 1e-12
        assert predict_formula(formula, delta_x_na=raw_dx)["DeltaV_pred"] == result["DeltaV_pred"]
        assert result["prediction_allowed"]
        results.append({"name": name, "Qch": qch, "DeltaXNa": raw_dx, "DeltaV_pred": result["DeltaV_pred"]})
    assert validate_delta_x_na(0.905, 0.905) == 0.905
    for value in (-0.1, float("nan"), float("inf"), 0.9051):
        try:
            validate_delta_x_na(value, 0.905)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid extraction was not blocked")
    mass = calculate_descriptors(parse_formula(_PRECISION_CASES[0][1]))["molar_mass"]
    try:
        predict_formula(_PRECISION_CASES[0][1], qch=0.93 * FARADAY / (mass * 3.6))
    except ValueError:
        pass
    else:
        raise AssertionError("Over-limit capacity was not blocked")
    phase_block = predict_formula("Na5/6Li5/18Mn13/18O2", delta_x_na=0.5)
    assert not phase_block["prediction_allowed"]
    return {"application_version": APP_VERSION, "passed": True, "ce_radius_A": ELEMENTS["Ce"].radius, "ce_descriptor_check": {"formula": ce_formula, "Ravg": ce_descriptors["Ravg"], "PhiTM": ce_descriptors["PhiTM"]}, "full_precision_cases": results, "physical_guards": "passed", "phase_gate": "passed"}


def _precision_gui_check(app, snapshot_path=None):
    from PySide6 import QtCore, QtGui, QtWidgets
    import tempfile
    import csv
    import openpyxl
    window = next(w for w in app.topLevelWidgets() if hasattr(w, "predict_tab"))
    tab = window.predict_tab
    assert tab.stack.currentIndex() == 1
    assert tab.formula_edit.text() == _PRECISION_CASES[0][1]
    assert tab.formula_mode_qch.isChecked() and not tab.formula_mode_dx.isChecked()
    assert tab.formula_qch.value() == 149.24
    assert tab.formula_dx.value() == predict_formula(_PRECISION_CASES[0][1], qch=149.24)["DeltaXNa"]
    startup_defaults = {"formula": tab.formula_edit.text(), "input_mode": "Qch", "Qch": tab.formula_qch.value(), "DeltaXNa": tab.formula_dx.value(), "formula_page_visible": True}
    assert "is used/displayed" not in window.model_info_tab.text.toPlainText()
    records = []
    for name, formula, qch, expected in _PRECISION_CASES:
        tab.formula_edit.setText(formula)
        tab.formula_mode_qch.setChecked(True)
        tab.formula_qch.setValue(qch)
        app.processEvents()
        assert tab.formula_qch.value() == qch
        raw_dx = predict_formula(formula, qch=qch)["DeltaXNa"]
        assert tab.formula_dx.value() == raw_dx
        tab.formula_dx.focusInEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusIn))
        tab.formula_dx.focusOutEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut))
        assert tab.formula_dx.value() == raw_dx
        assert tab.btn_formula.isEnabled()
        tab.run_formula()
        assert abs(tab.comparison_items[-1]["DeltaV_pred"] - expected) < 1e-12
        # A mode change uses the full linked value, even though idle text shows 2 decimals.
        tab.formula_mode_dx.setChecked(True)
        tab.run_formula()
        assert abs(tab.comparison_items[-1]["DeltaV_pred"] - expected) < 1e-12
        records.append({"name": name, "dx_internal": tab.formula_dx.value(), "dx_display": tab.formula_dx.textFromValue(tab.formula_dx.value()), "prediction": expected})
    # Commit a real text edit, and then blur/refocus without changing it.
    box = tab.formula_qch
    box.focusInEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusIn))
    box.lineEdit().setText("149.24")
    box.lineEdit().textEdited.emit("149.24")
    box.interpretText()
    box.focusOutEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut))
    assert box.value() == 149.24 and box.lineEdit().text() == "149.24"
    box.focusInEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusIn))
    box.focusOutEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut))
    assert box.value() == 149.24
    box.focusInEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusIn))
    box.lineEdit().setText("149.246789")
    box.lineEdit().textEdited.emit("149.246789")
    box.interpretText()
    box.focusOutEvent(QtGui.QFocusEvent(QtCore.QEvent.Type.FocusOut))
    assert box.value() == 149.246789 and box.lineEdit().text() == "149.25"
    small_values = []
    tiny_box = QtWidgets.QDoubleSpinBox()
    tiny_box.valueChanged.connect(small_values.append)
    tiny_box.setValue(1.0000000000000002e-5)
    tiny_box.setValue(1.0000000000000004e-5)
    assert len(small_values) == 2 and tiny_box.value() == 1.0000000000000004e-5
    # Element designer must retain three-decimal stoichiometries too.
    composition = parse_formula(_PRECISION_CASES[2][1])
    for element, checkbox in tab.selector.checkboxes.items():
        checkbox.setChecked(element in composition)
    for element, widget in tab.element_inputs.items():
        widget.setValue(composition.get(element, 0.0))
    tab.na_spin.setValue(composition["Na"])
    tab.o_spin.setValue(composition["O"])
    tab.des_mode_qch.setChecked(True)
    tab.des_qch.setValue(140.6)
    assert tab.na_spin.value() == 0.905
    assert tab.element_inputs["Li"].value() == 0.015
    assert tab.element_inputs["Mo"].value() == 0.015
    tab.run_designer()
    assert abs(tab.comparison_items[-1]["DeltaV_pred"] - _PRECISION_CASES[2][3]) < 1e-12
    tab.des_dx.setValue(0.906)
    tab.des_mode_dx.setChecked(True)
    assert not tab.btn_designer.isEnabled()
    tab.des_dx.setValue(0.905)
    assert tab.btn_designer.isEnabled()
    # Exercise the actual GUI batch loader and export methods with file dialogs
    # replaced only during this self-test; avoid interactive prompts.
    with tempfile.TemporaryDirectory(prefix="emvrf_precision_check_") as scratch:
        scratch = Path(scratch)
        fixture = scratch / "inputs.csv"
        pd.DataFrame([{"Name": n, "Formula": f, "Q_ch": q} for n, f, q, _ in _PRECISION_CASES]).to_csv(fixture, index=False)
        original_open = QtWidgets.QFileDialog.getOpenFileName
        original_save = QtWidgets.QFileDialog.getSaveFileName
        original_critical = QtWidgets.QMessageBox.critical
        def fail_dialog(*args):
            raise AssertionError("GUI error dialog: " + str(args[-1]))
        try:
            QtWidgets.QMessageBox.critical = staticmethod(fail_dialog)
            QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *args: (str(fixture), "CSV (*.csv)"))
            tab.batch_load_file()
            assert len(tab.batch_df) == 4
            # Locate the numeric result by its values, preserving existing headers.
            delta_column = next(c for c in tab.batch_df.columns if all(isinstance(v, (int, float, np.floating)) and abs(float(v) - row[3]) < 1e-12 for v, row in zip(tab.batch_df[c], _PRECISION_CASES)))
            csv_path = scratch / "export.csv"
            QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *args: (str(csv_path), "CSV (*.csv)"))
            tab.batch_export_file()
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            assert all(float(row[delta_column]) == expected[3] for row, expected in zip(rows, _PRECISION_CASES))
            xlsx_base = scratch / "export_no_extension"
            QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *args: (str(xlsx_base), "Excel (*.xlsx)"))
            tab.batch_export_file()
            workbook = openpyxl.load_workbook(str(xlsx_base) + ".xlsx", data_only=True)
            sheet = workbook.active
            column = next(cell.column for cell in sheet[1] if cell.value == delta_column)
            for index, expected in enumerate(_PRECISION_CASES, start=2):
                cell = sheet.cell(index, column)
                assert abs(cell.value - expected[3]) < 1e-14 and cell.number_format == "0.00"
            workbook.close()
        finally:
            QtWidgets.QFileDialog.getOpenFileName = original_open
            QtWidgets.QFileDialog.getSaveFileName = original_save
            QtWidgets.QMessageBox.critical = original_critical
    tab.formula_edit.setText(_PRECISION_CASES[0][1])
    tab.formula_mode_qch.setChecked(True)
    tab.formula_qch.setValue(149.24)
    tab.formula_edit.setFocus()
    app.processEvents()
    if snapshot_path:
        if not window.grab().save(str(snapshot_path)):
            raise RuntimeError("Unable to save GUI snapshot")
    result = {"application_version": APP_VERSION, "gui_passed": True, "startup_defaults": startup_defaults, "formula_and_mode_switch_cases": records, "text_edit_and_focus": "passed", "adjacent_float_signal": "passed", "designer_fractional_stoichiometry": "passed", "gui_physical_guard": "passed", "batch_import_and_csv_xlsx_export": "passed"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_gui(smoke_test=False, snapshot_path=None):
    from PySide6 import QtWidgets
    original_spin = QtWidgets.QDoubleSpinBox
    original_app = QtWidgets.QApplication
    QtWidgets.QDoubleSpinBox = _make_precision_spinbox(original_spin)
    class PrecisionApplication(original_app):
        def exec(self):
            window = next(w for w in self.topLevelWidgets() if hasattr(w, "predict_tab"))
            tab = window.predict_tab
            tab.formula_edit.setText(_PRECISION_CASES[0][1])
            tab.formula_mode_qch.setChecked(True)
            tab.formula_qch.setValue(149.24)
            tab._sync_formula_from_qch()
            tab.stack.setCurrentIndex(1)
            window.predict_tab._sync_designer_from_dx()
            if smoke_test or snapshot_path:
                result = _precision_gui_check(self, snapshot_path)
                if snapshot_path:
                    Path(str(snapshot_path) + ".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                for widget in self.topLevelWidgets():
                    widget.close()
                return 0
            return super().exec()
    QtWidgets.QApplication = PrecisionApplication
    try:
        return _original_run_gui(smoke_test=False, snapshot_path=None)
    finally:
        QtWidgets.QDoubleSpinBox = original_spin
        QtWidgets.QApplication = original_app


# ============================================================================
# Revision 2026-10-08: numeric-field steps on the decimal grid.
# Wheel/arrow steps no longer accumulate binary floating-point error.
# ============================================================================
"""EM-VRF spin-box step fix (2026-10-08). Executed in the application globals after
full_precision_patch.py.

FullPrecisionSpinBox.stepBy added wheel/arrow/PageUp steps in binary floating
point, so repeated steps accumulated representation error that the full-precision
design then kept (0.05 + 0.01 -> 0.060000000000000005, 0.09 + 0.01 ->
0.09999999999999999). Steps are now taken on the decimal grid of the current value
and the step size, so 0.09 + 0.01 -> 0.1 while typed full-precision values are
kept (0.0153 + 0.01 -> 0.0253). Typed input, linked values and the model are
unchanged.
"""
from decimal import Decimal as _StepDecimal

_unpatched_make_precision_spinbox = _make_precision_spinbox


def _decimal_step_by(self, steps):
    value = _StepDecimal(repr(float(self.value()))) + int(steps) * _StepDecimal(repr(float(self.singleStep())))
    self.setValue(float(value))


def _make_precision_spinbox(base):
    spinbox = _unpatched_make_precision_spinbox(base)
    spinbox.stepBy = _decimal_step_by
    return spinbox


# ============================================================================
# Revision 2026-10-09: modern interface (presentation only).
# Themes, layout, latest-prediction card, comparison Table/Chart switch.
# ============================================================================
"""EM-VRF modern UI layer. Executed in the original application globals after
full_precision_patch.py.

Presentation only: the model, descriptors, phase gate, precision handling,
batch import/export and every displayed number format are unchanged. The layer
re-themes the widgets that run_gui() builds (light/dark, following Windows),
adds a latest-prediction summary card, and repaints the comparison chart with
theme colours. Any failure here falls back to the original look.
"""
import html as _ui_html
import math as _ui_math
import os as _ui_os
import re as _ui_re
import sys as _ui_sys
import tempfile as _ui_tempfile
import traceback as _ui_traceback
from pathlib import Path as _UIPath
from string import Template as _UITemplate

UI_REVISION = "2026.10.09-modern-ui-r3"
UI_THEME_ENV = "EMVRF_THEME"
_precision_run_gui = run_gui
_UI = {"controller": None}

_UI_THEMES = {
    "light": {
        "bg": "#F3F5F7", "surface": "#FFFFFF", "surface_alt": "#F6F8FA", "input": "#FFFFFF", "row_alt": "#F8FAFB",
        "border": "#E1E6EB", "border_strong": "#CED5DC", "divider": "#EDF0F3",
        "text": "#15202B", "muted": "#566370", "faint": "#8C96A1",
        "hover": "#EDF1F4", "pressed": "#E2E8ED",
        "accent": "#0E7C86", "accent_hover": "#0B6B74", "accent_pressed": "#095A62", "on_accent": "#FFFFFF",
        "accent_soft": "#DDF0F1", "accent_text": "#0A6A72",
        "success": "#1C7A43", "success_bg": "#E5F4EB", "success_border": "#BCE1CB",
        "warning": "#995A00", "warning_bg": "#FFF4DE", "warning_border": "#F0D29A",
        "danger": "#B42318", "danger_bg": "#FDECEA", "danger_border": "#F4C3BD",
        "scroll": "#C8D0D7", "scroll_hover": "#A9B3BD", "tooltip_bg": "#1E2933", "tooltip_text": "#F4F7FA",
        "disabled_bg": "#E7ECF0",
        "chart_line": "#0E7C86", "chart_low": "#2B5C8A", "chart_high": "#D9822B", "chart_grid": "#EBEFF2",
        "chart_axis": "#C5CCD3", "chart_zero": "#97A2AD", "chart_err": "#A7B1BB", "chart_ring": "#FFFFFF",
    },
    "dark": {
        "bg": "#0E1216", "surface": "#161B20", "surface_alt": "#1B2127", "input": "#11161B", "row_alt": "#191F25",
        "border": "#252C34", "border_strong": "#323A44", "divider": "#1F252C",
        "text": "#E4E8EC", "muted": "#99A4AF", "faint": "#66717C",
        "hover": "#20272E", "pressed": "#29313A",
        "accent": "#36BBB3", "accent_hover": "#4CC8C0", "accent_pressed": "#2A9F98", "on_accent": "#03201E",
        "accent_soft": "#123638", "accent_text": "#5CD2CA",
        "success": "#5BC88A", "success_bg": "#122C1D", "success_border": "#1E4A31",
        "warning": "#EFB256", "warning_bg": "#30240E", "warning_border": "#574116",
        "danger": "#F2847B", "danger_bg": "#361513", "danger_border": "#5C241F",
        "scroll": "#37404A", "scroll_hover": "#4A5560", "tooltip_bg": "#E4E8EC", "tooltip_text": "#11161B",
        "disabled_bg": "#1D2329",
        "chart_line": "#36BBB3", "chart_low": "#6EA8DC", "chart_high": "#F0A04B", "chart_grid": "#20272E",
        "chart_axis": "#39424C", "chart_zero": "#6A7580", "chart_err": "#55606B", "chart_ring": "#161B20",
    },
}

_UI_QSS = """
QMainWindow, QDialog { background: $bg; }
QWidget { color: $text; }
QLabel { background: transparent; }
QToolTip { background: $tooltip_bg; color: $tooltip_text; border: 0px; padding: 6px 8px; }

QTabWidget#mainTabs::pane { border: 0px; background: $bg; top: 0px; }
QTabWidget#mainTabs > QTabBar { background: transparent; font-weight: 600; }
QTabWidget#mainTabs > QTabBar::tab { background: transparent; color: $muted; border: 1px solid transparent;
    border-radius: 8px; padding: 6px 16px; margin: 11px 4px 11px 0px; }
QTabWidget#mainTabs > QTabBar::tab:hover { background: $hover; color: $text; }
QTabWidget#mainTabs > QTabBar::tab:selected { background: $surface; color: $text; border: 1px solid $border; }
QLabel#brandName { font-size: 12pt; font-weight: 600; color: $text; }
QFrame#brandDivider { background: $border; border: 0px; }
QLabel#versionChip { color: $muted; background: $surface; border: 1px solid $border; border-radius: 11px;
    padding: 2px 10px; font-size: 9pt; }
QToolButton#themeButton { background: $surface; color: $text; border: 1px solid $border; border-radius: 8px;
    padding: 4px 10px; font-weight: 600; }
QToolButton#themeButton:hover { background: $hover; }
QToolButton#themeButton::menu-indicator { image: none; width: 0px; }
QMenu { background: $surface; border: 1px solid $border_strong; padding: 4px; }
QMenu::item { padding: 6px 28px 6px 12px; border-radius: 6px; color: $text; background: transparent; }
QMenu::item:selected { background: $hover; }
QMenu::item:checked { color: $accent_text; font-weight: 600; }
QMenu::indicator { width: 0px; height: 0px; }

QGroupBox { background: $surface; border: 1px solid $border; border-radius: 12px; margin-top: 0px;
    padding: 40px 12px 12px 12px; font-size: 11pt; font-weight: 600; }
QGroupBox::title { subcontrol-origin: padding; subcontrol-position: top left; left: 16px; top: 14px;
    padding: 0px; color: $text; background: transparent; }
QGroupBox#subGroup { background: transparent; border: 0px; border-radius: 0px; padding: 28px 0px 0px 0px;
    font-size: 10pt; }
QGroupBox#subGroup::title { left: 0px; top: 4px; color: $muted; }
QFrame[card="true"] { background: $surface; border: 1px solid $border; border-radius: 12px; }
QFrame#heroCard { background: $surface; border: 1px solid $border; border-radius: 12px; }
QLabel#cardTitle { font-size: 11pt; font-weight: 600; color: $text; }
QLabel#pageTitle { font-size: 18pt; font-weight: 600; color: $text; }
QLabel#pageSubtitle { color: $muted; font-size: 10pt; }
QLabel#fileLabel { color: $muted; }

QLabel#eyebrow { color: $faint; font-size: 8pt; font-weight: 700; }
QLabel#heroTitle { color: $text; }
QLabel#heroValue { color: $text; font-size: 26pt; font-weight: 600; }
QLabel#heroCaption { color: $muted; }
QLabel#statLabel { color: $faint; font-size: 9pt; }
QLabel#statValue { color: $text; font-weight: 600; }
QLabel[chip="neutral"] { color: $muted; background: $surface_alt; border: 1px solid $border; border-radius: 10px;
    padding: 1px 9px; font-size: 9pt; font-weight: 600; }
QLabel[chip="success"] { color: $success; background: $success_bg; border: 1px solid $success_border;
    border-radius: 10px; padding: 1px 9px; font-size: 9pt; font-weight: 600; }
QLabel[chip="warning"] { color: $warning; background: $warning_bg; border: 1px solid $warning_border;
    border-radius: 10px; padding: 1px 9px; font-size: 9pt; font-weight: 600; }
QLabel[chip="danger"] { color: $danger; background: $danger_bg; border: 1px solid $danger_border;
    border-radius: 10px; padding: 1px 9px; font-size: 9pt; font-weight: 600; }

QLineEdit, QAbstractSpinBox { background: $input; color: $text; border: 1px solid $border_strong; border-radius: 8px;
    padding: 5px 10px; min-height: 22px; selection-background-color: $accent; selection-color: $on_accent; }
QLineEdit:hover, QAbstractSpinBox:hover { border-color: $faint; }
QLineEdit:focus, QAbstractSpinBox:focus { border-color: $accent; }
QLineEdit:disabled, QAbstractSpinBox:disabled { color: $faint; background: $surface_alt; border-color: $border; }
QAbstractSpinBox { padding-right: 30px; }
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 22px; border: 0px;
    background: transparent; margin-right: 4px; }
QAbstractSpinBox::up-button { subcontrol-position: top right; margin-top: 3px; }
QAbstractSpinBox::down-button { subcontrol-position: bottom right; margin-bottom: 3px; }
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: $hover; border-radius: 4px; }

QPushButton { background: $surface; color: $text; border: 1px solid $border_strong; border-radius: 8px;
    padding: 6px 14px; font-weight: 600; min-height: 20px; }
QPushButton:hover { background: $hover; }
QPushButton:pressed { background: $pressed; }
QPushButton:focus { border-color: $accent; }
QPushButton:disabled { color: $faint; background: $surface_alt; border-color: $border; }
QPushButton[variant="primary"] { background: $accent; color: $on_accent; border: 1px solid $accent; padding: 8px 18px; }
QPushButton[variant="primary"]:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton[variant="primary"]:pressed { background: $accent_pressed; border-color: $accent_pressed; }
QPushButton[variant="primary"]:disabled { background: $disabled_bg; border-color: $disabled_bg; color: $faint; }
QPushButton[variant="ghost"] { background: transparent; border: 1px solid transparent; color: $muted; padding: 5px 10px; }
QPushButton[variant="ghost"]:hover { background: $hover; color: $text; }
QPushButton[variant="ghost"]:pressed { background: $pressed; }
QPushButton[variant="card"] { background: $surface; border: 1px solid $border; border-radius: 14px; padding: 0px; }
QPushButton[variant="card"]:hover { border-color: $accent; }
QPushButton[variant="card"]:pressed { background: $accent_soft; }
QFrame#segmented { background: $surface_alt; border: 1px solid $border; border-radius: 9px; }
QFrame#segmented QPushButton { background: transparent; border: 1px solid transparent; border-radius: 7px;
    color: $muted; padding: 3px 14px; min-height: 18px; font-weight: 600; }
QFrame#segmented QPushButton:hover { color: $text; }
QFrame#segmented QPushButton:checked { background: $surface; border-color: $border_strong; color: $text; }
QLabel#modeCardTitle { font-size: 12pt; font-weight: 600; color: $text; }
QLabel#modeCardText { color: $muted; }
QLabel#modeCardIcon { background: $accent_soft; border-radius: 12px; }
QLabel#modeCardArrow { color: $faint; font-size: 16pt; }

QRadioButton, QCheckBox { spacing: 8px; background: transparent; }

QWidget#periodicSelector QCheckBox { background: $surface_alt; border: 1px solid $border; border-radius: 7px;
    padding: 0px; font-weight: 600; spacing: 0px; }
QWidget#periodicSelector QCheckBox::indicator { width: 0px; height: 0px; image: none; }
QWidget#periodicSelector QCheckBox:hover { border-color: $accent; }
QWidget#periodicSelector QCheckBox:checked { background: $accent; border-color: $accent; color: $on_accent; }
QWidget#periodicSelector QCheckBox:disabled { background: transparent; border: 1px dashed $border; color: $faint; }

QTableView { background: $surface; alternate-background-color: $row_alt; border: 0px; gridline-color: $divider;
    selection-background-color: $accent_soft; selection-color: $text; }
QTableView::item { padding: 0px 8px; border: 0px; border-bottom: 1px solid $divider; }
QTableView::item:selected { background: $accent_soft; color: $text; }
QHeaderView { background: $surface; border: 0px; color: $muted; }
QHeaderView::section { background: $surface_alt; color: $muted; border: 0px; border-bottom: 1px solid $border;
    padding: 4px 8px; font-weight: 600; }
QHeaderView::section:vertical { background: $surface; color: $faint; border: 0px; border-bottom: 1px solid $divider;
    padding: 0px 8px; font-weight: 400; }
QTableCornerButton::section { background: $surface_alt; border: 0px; border-bottom: 1px solid $border; }

QTextEdit { background: $surface; color: $text; border: 0px; selection-background-color: $accent;
    selection-color: $on_accent; }
QTextEdit#docView { border: 1px solid $border; border-radius: 12px; }

QScrollArea { background: transparent; border: 0px; }
QScrollArea#chartScroll, QScrollArea#chartScroll > QWidget { background: $surface; }
QSplitter::handle { background: transparent; }
QScrollBar:vertical { background: transparent; width: 12px; margin: 2px; }
QScrollBar::handle:vertical { background: $scroll; border-radius: 4px; min-height: 36px; margin: 0px 2px; }
QScrollBar:horizontal { background: transparent; height: 12px; margin: 2px; }
QScrollBar::handle:horizontal { background: $scroll; border-radius: 4px; min-width: 36px; margin: 2px 0px; }
QScrollBar::handle:hover { background: $scroll_hover; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; border: 0px; background: transparent; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QAbstractScrollArea::corner { background: transparent; }
"""

_UI_QSS_ICONS = """
QAbstractSpinBox::up-arrow { image: url("$icons/chev_up.svg"); width: 10px; height: 10px; }
QAbstractSpinBox::down-arrow { image: url("$icons/chev_down.svg"); width: 10px; height: 10px; }
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off { image: url("$icons/chev_up_off.svg"); }
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off { image: url("$icons/chev_down_off.svg"); }
QRadioButton::indicator, QCheckBox::indicator { width: 18px; height: 18px; }
QRadioButton::indicator:unchecked { image: url("$icons/radio_off.svg"); }
QRadioButton::indicator:unchecked:hover { image: url("$icons/radio_hover.svg"); }
QRadioButton::indicator:checked { image: url("$icons/radio_on.svg"); }
QCheckBox::indicator:unchecked { image: url("$icons/check_off.svg"); }
QCheckBox::indicator:unchecked:hover { image: url("$icons/check_hover.svg"); }
QCheckBox::indicator:checked { image: url("$icons/check_on.svg"); }
QRadioButton::indicator:disabled { image: url("$icons/radio_disabled.svg"); }
QCheckBox::indicator:disabled { image: url("$icons/check_disabled.svg"); }
"""

# Inline colours used by the original widgets, mapped to theme roles.
_UI_LEGACY_ROLES = {"#666d75": "muted", "gray": "muted", "grey": "muted", "green": "success", "#2e7d32": "success",
                    "red": "danger", "#b3261e": "danger", "#a15c00": "warning"}


def _ui_report(context):
    try:
        print("[modern-ui] " + context + " failed; the original styling is kept for that part.", file=_ui_sys.stderr)
        _ui_traceback.print_exc()
    except Exception:
        pass


def _ui_tokens():
    controller = _UI["controller"]
    return controller.tokens if controller is not None else _UI_THEMES["light"]


def _ui_icon_svgs(t):
    def svg(size, body):
        return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">%s</svg>'
                % (size, size, size, size, body))

    def chevron(path, colour):
        return svg(12, '<path d="%s" fill="none" stroke="%s" stroke-width="1.7" stroke-linecap="round" '
                       'stroke-linejoin="round"/>' % (path, colour))

    def radio(stroke, fill):
        return svg(18, '<circle cx="9" cy="9" r="7.25" fill="%s" stroke="%s" stroke-width="1.5"/>' % (fill, stroke))

    def check(stroke, fill):
        return svg(18, '<rect x="1.75" y="1.75" width="14.5" height="14.5" rx="4" fill="%s" stroke="%s" '
                       'stroke-width="1.5"/>' % (fill, stroke))

    line = 'fill="none" stroke="%s" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"'
    rays = "".join('<path d="M%.2f %.2fL%.2f %.2f" %s/>' % (
        8 + 5.2 * _ui_math.cos(a), 8 + 5.2 * _ui_math.sin(a), 8 + 6.8 * _ui_math.cos(a), 8 + 6.8 * _ui_math.sin(a),
        line % t["text"]) for a in [i * _ui_math.pi / 4 for i in range(8)])
    accent = t["accent_text"]
    return {
        "chev_up": chevron("M3 7.5 6 4.5 9 7.5", t["muted"]),
        "chev_down": chevron("M3 4.5 6 7.5 9 4.5", t["muted"]),
        "chev_up_off": chevron("M3 7.5 6 4.5 9 7.5", t["border_strong"]),
        "chev_down_off": chevron("M3 4.5 6 7.5 9 4.5", t["border_strong"]),
        "radio_off": radio(t["border_strong"], t["input"]),
        "radio_hover": radio(t["accent"], t["input"]),
        "radio_disabled": radio(t["border"], t["surface_alt"]),
        "radio_on": svg(18, '<circle cx="9" cy="9" r="8" fill="%s"/><circle cx="9" cy="9" r="3.2" fill="%s"/>'
                            % (t["accent"], t["on_accent"])),
        "check_off": check(t["border_strong"], t["input"]),
        "check_hover": check(t["accent"], t["input"]),
        "check_disabled": check(t["border"], t["surface_alt"]),
        "check_on": svg(18, '<rect x="1" y="1" width="16" height="16" rx="4.5" fill="%s"/><path d="M5 9.2 7.8 12 '
                            '13 6.5" fill="none" stroke="%s" stroke-width="2" stroke-linecap="round" '
                            'stroke-linejoin="round"/>' % (t["accent"], t["on_accent"])),
        "theme_light": svg(16, '<circle cx="8" cy="8" r="3.2" %s/>%s' % (line % t["text"], rays)),
        "theme_dark": svg(16, '<path d="M13.2 10.1A5.6 5.6 0 0 1 5.9 2.8a5.6 5.6 0 1 0 7.3 7.3z" %s/>'
                              % (line % t["text"])),
        "theme_system": svg(16, '<circle cx="8" cy="8" r="5.6" %s/><path d="M8 2.4a5.6 5.6 0 0 1 0 11.2z" fill="%s"/>'
                                % (line % t["text"], t["text"])),
        "mode_formula": svg(24, '<rect x="3.5" y="4.5" width="17" height="15" rx="3" %s/><path d="M7.5 9.5h5M7.5 '
                                '13h9M7.5 16.5h6" %s/><path d="M16 8v3" %s/>'
                                % (line % accent, line % accent, line % accent)),
        "mode_designer": svg(24, '<rect x="3.5" y="3.5" width="7.5" height="7.5" rx="2" fill="%s"/><rect x="13" '
                                 'y="3.5" width="7.5" height="7.5" rx="2" %s/><rect x="3.5" y="13" width="7.5" '
                                 'height="7.5" rx="2" %s/><rect x="13" y="13" width="7.5" height="7.5" rx="2" %s/>'
                                 % (accent, line % accent, line % accent, line % accent)),
    }


def _ui_write_icons(name, t):
    try:
        root = _UIPath(_ui_tempfile.gettempdir()) / "EM-VRF-modern-ui" / (UI_REVISION + "-" + name)
        root.mkdir(parents=True, exist_ok=True)
        for key, svg in _ui_icon_svgs(t).items():
            target = root / (key + ".svg")
            if not target.is_file() or target.read_text(encoding="utf-8") != svg:
                target.write_text(svg, encoding="utf-8")
        return root.as_posix()
    except Exception:
        _ui_report("icon cache")
        return None


def _ui_pick_font():
    from PySide6.QtGui import QFont, QFontDatabase
    available = set(QFontDatabase.families())
    families = [f for f in ("Segoe UI Variable Text", "Segoe UI", "Arial") if f in available]
    font = QFont()
    font.setFamilies(families + ["Microsoft YaHei UI", "Microsoft YaHei"])
    font.setPointSize(10)
    return font


def _ui_palette(t):
    from PySide6.QtGui import QColor, QPalette
    role, group = QPalette.ColorRole, QPalette.ColorGroup
    palette = QPalette()
    for key, value in ((role.Window, "bg"), (role.WindowText, "text"), (role.Base, "input"),
                       (role.AlternateBase, "row_alt"), (role.Text, "text"), (role.Button, "surface"),
                       (role.ButtonText, "text"), (role.BrightText, "on_accent"), (role.Highlight, "accent"),
                       (role.HighlightedText, "on_accent"), (role.ToolTipBase, "tooltip_bg"),
                       (role.ToolTipText, "tooltip_text"), (role.PlaceholderText, "faint"), (role.Link, "accent_text"),
                       (role.LinkVisited, "accent_text"), (role.Light, "surface"), (role.Midlight, "border"),
                       (role.Mid, "border_strong"), (role.Dark, "border_strong")):
        palette.setColor(key, QColor(t[value]))
    for key in (role.WindowText, role.Text, role.ButtonText):
        palette.setColor(group.Disabled, key, QColor(t["faint"]))
    palette.setColor(group.Disabled, role.Base, QColor(t["surface_alt"]))
    if hasattr(role, "Accent"):
        palette.setColor(role.Accent, QColor(t["accent"]))
    return palette


def _ui_stylesheet(t, icon_dir):
    qss = _UITemplate(_UI_QSS).substitute(t)
    if icon_dir:
        qss += _UITemplate(_UI_QSS_ICONS).substitute(icons=icon_dir)
    return qss


def _ui_translate_css(css, t):
    """Map the original inline label styles onto the active theme."""
    text = str(css or "").strip()
    if not text or "{" in text:
        return text
    props = {}
    for part in text.split(";"):
        if ":" in part:
            key, value = part.split(":", 1)
            props[key.strip().lower()] = value.strip()
    role = _UI_LEGACY_ROLES.get(props.get("color", "").lower())
    bold = props.get("font-weight", "").lower() in ("bold", "600", "700")
    if role in ("success", "danger", "warning") and bold:
        colours = (t[role], t[role + "_bg"], t[role + "_border"])
        if "font-size" in props:
            return "color: %s; background: %s; border: 1px solid %s; border-radius: 8px; padding: 7px 10px; " \
                   "font-weight: 600;" % colours
        return "color: %s; background: %s; border: 1px solid %s; border-radius: 6px; padding: 1px 8px; " \
               "font-weight: 600;" % colours
    if role == "muted":
        return "color: %s; font-size: 9pt;" % t["muted"]
    if role:
        return "color: %s;%s" % (t[role], " font-weight: 600;" if bold else "")
    return text


def _ui_formula_html(formula):
    """Chemical formula with subscripted stoichiometry, for display only (wraps between elements)."""
    parts = _ui_re.findall(r"([A-Z][a-z]?)(\d*\.?\d+(?:/\d+)?)?", formula)
    if not parts or "".join(e + n for e, n in parts) != formula:
        return _ui_html.escape(formula)
    return "&#8203;".join(_ui_html.escape(e) + ("<sub>%s</sub>" % _ui_html.escape(n) if n else "") for e, n in parts)


def _ui_doc_css(t, family):
    return ("<style>\n"
            "body { font-family: '%(family)s'; color: %(text)s; line-height: 135%%; }\n"
            "h2 { color: %(text)s; font-size: 17pt; font-weight: 600; margin-top: 0px; margin-bottom: 6px; }\n"
            "h3 { color: %(accent_text)s; font-size: 12pt; font-weight: 600; margin-top: 18px; margin-bottom: 6px; }\n"
            "table { border-collapse: collapse; margin: 6px 0 12px 0; }\n"
            "th, td { border: 1px solid %(border)s; padding: 6px 10px; }\n"
            "th { background: %(surface_alt)s; color: %(muted)s; font-weight: 600; }\n"
            ".note { background: %(warning_bg)s; border: 1px solid %(warning_border)s; padding: 10px; }\n"
            "</style>") % dict(t, family=family)


def _ui_nice_step(raw):
    if not raw > 0 or not _ui_math.isfinite(raw):
        return 1.0
    exponent = _ui_math.floor(_ui_math.log10(raw))
    scale = 10.0 ** exponent
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= multiple * scale * (1 + 1e-9):
            return multiple * scale
    return 10.0 * scale


def _ui_blend(start, end, fraction):
    from PySide6.QtGui import QColor
    fraction = max(0.0, min(1.0, fraction))
    return QColor(round(start.red() + (end.red() - start.red()) * fraction),
                  round(start.green() + (end.green() - start.green()) * fraction),
                  round(start.blue() + (end.blue() - start.blue()) * fraction))


def _ui_chart_paint(self, event):
    """Theme-aware repaint of DeltaVComparisonChart; same data, interval bars and labels."""
    from PySide6.QtCore import QPointF, QRectF, Qt
    from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
    t = _ui_tokens()
    align = Qt.AlignmentFlag
    painter = QPainter(self)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(t["surface"]))
        outer = QRectF(self.rect()).adjusted(6, 6, -6, -6)
        base_font = QFont(self.font())
        if not self.items:
            painter.setPen(QColor(t["faint"]))
            painter.setFont(base_font)
            painter.drawText(outer, align.AlignCenter, "Run two or more predictions to compare signed ΔV")
            return
        values = [float(item["DeltaV_pred"]) for item in self.items]
        lower = [float(item["DeltaV_lower"]) for item in self.items]
        upper = [float(item["DeltaV_upper"]) for item in self.items]
        y_min = min(min(values + lower), 0.0)
        y_max = max(max(values + upper), 0.0)
        span = max(y_max - y_min, 0.5)
        y_min -= 0.08 * span
        y_max += 0.08 * span
        step = _ui_nice_step((y_max - y_min) / 5.0)
        y_min = _ui_math.floor(y_min / step + 1e-9) * step
        y_max = _ui_math.ceil(y_max / step - 1e-9) * step
        decimals = 0 if abs(step - round(step)) < 1e-9 else (1 if abs(step * 10 - round(step * 10)) < 1e-9 else 2)
        ticks = [y_min + index * step for index in range(int(round((y_max - y_min) / step)) + 1)]
        small = QFont(base_font)
        small.setPointSizeF(max(7.5, base_font.pointSizeF() - 1))
        metrics = QFontMetricsF(small)
        tick_labels = [format(0.0 if abs(v) < step * 1e-6 else v, ".%df" % decimals) for v in ticks]
        tick_width = max(metrics.horizontalAdvance(label) for label in tick_labels)
        plot = QRectF(outer.left() + 30 + tick_width + 10, outer.top() + 34, 0, 0)
        plot.setRight(outer.right() - 12)
        plot.setBottom(outer.bottom() - 56)

        def y_at(value):
            return plot.bottom() - (value - y_min) / (y_max - y_min) * plot.height()

        count = len(values)
        pad = min(46.0, plot.width() / (2.0 * count))
        x_step = (plot.width() - 2 * pad) / max(count - 1, 1)
        xs = [plot.center().x()] if count == 1 else [plot.left() + pad + index * x_step for index in range(count)]

        painter.setFont(small)
        for value, label in zip(ticks, tick_labels):
            y = y_at(value)
            painter.setPen(QPen(QColor(t["chart_grid"]), 1))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(t["muted"]))
            painter.drawText(QRectF(plot.left() - tick_width - 10, y - 9, tick_width, 18),
                             align.AlignRight | align.AlignVCenter, label)
        painter.setPen(QPen(QColor(t["chart_axis"]), 1))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        zero_y = y_at(0.0)
        painter.setPen(QPen(QColor(t["chart_zero"]), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(plot.left(), zero_y), QPointF(plot.right(), zero_y))

        error_pen = QPen(QColor(t["chart_err"]), 1.4)
        error_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(error_pen)
        for x, low, high in zip(xs, lower, upper):
            low_y, high_y = y_at(low), y_at(high)
            painter.drawLine(QPointF(x, low_y), QPointF(x, high_y))
            painter.drawLine(QPointF(x - 5, low_y), QPointF(x + 5, low_y))
            painter.drawLine(QPointF(x - 5, high_y), QPointF(x + 5, high_y))

        if count > 1:
            path = QPainterPath(QPointF(xs[0], y_at(values[0])))
            for index in range(1, count):
                midpoint = (xs[index - 1] + xs[index]) / 2
                path.cubicTo(QPointF(midpoint, y_at(values[index - 1])), QPointF(midpoint, y_at(values[index])),
                             QPointF(xs[index], y_at(values[index])))
            line_colour = QColor(t["chart_line"])
            line_colour.setAlpha(200)
            line_pen = QPen(line_colour, 2.4)
            line_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(line_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

        low_colour, mid_colour, high_colour = QColor(t["chart_low"]), QColor(t["chart_line"]), QColor(t["chart_high"])

        def point_colour(fraction):
            # Low -> accent -> high avoids the muddy midpoint of a direct blue/orange blend.
            if fraction <= 0.5:
                return _ui_blend(low_colour, mid_colour, fraction * 2)
            return _ui_blend(mid_colour, high_colour, fraction * 2 - 1)

        value_span = max(max(values) - min(values), 1e-12)
        label_font = QFont(small)
        label_font.setWeight(QFont.Weight.DemiBold)
        label_metrics = QFontMetricsF(label_font)
        pill_fill = QColor(t["surface"])
        pill_fill.setAlpha(235)
        name_width = max(60.0, x_step if count > 1 else plot.width())
        for x, value, item in zip(xs, values, self.items):
            y = y_at(value)
            painter.setPen(QPen(QColor(t["chart_ring"]), 2.0))
            painter.setBrush(point_colour((value - min(values)) / value_span))
            painter.drawEllipse(QPointF(x, y), 6.5, 6.5)
            text = format(value, "+.2f") + "%"
            width, height = label_metrics.horizontalAdvance(text) + 14, label_metrics.height() + 4
            box = QRectF(x - width / 2, y - 14 - height, width, height)
            if box.top() < outer.top():
                box.moveTop(y + 12)
            box.translate(max(0.0, outer.left() - box.left()) + min(0.0, outer.right() - box.right()), 0)
            painter.setPen(QPen(QColor(t["border"]), 1))
            painter.setBrush(pill_fill)
            painter.drawRoundedRect(box, height / 2, height / 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setFont(label_font)
            painter.setPen(QColor(t["text"]))
            painter.drawText(box, align.AlignCenter, text)
            painter.setFont(small)
            painter.setPen(QColor(t["muted"]))
            name = metrics.elidedText(str(item["candidate"]), Qt.TextElideMode.ElideRight, name_width - 6)
            painter.drawText(QRectF(x - name_width / 2, plot.bottom() + 8, name_width, 18),
                             align.AlignHCenter | align.AlignTop, name)

        painter.setFont(small)
        painter.setPen(QColor(t["faint"]))
        painter.drawText(QRectF(plot.left(), outer.bottom() - 22, plot.width(), 20), align.AlignCenter,
                         "Candidate compositions")
        painter.save()
        painter.translate(outer.left() + 10, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot.height() / 2, -10, plot.height(), 20), align.AlignCenter, self.y_axis_label)
        painter.restore()
    finally:
        painter.end()


def _ui_widget_classes():
    from PySide6.QtCore import QEvent, QObject, Qt
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QSizePolicy, QVBoxLayout

    class PolishFilter(QObject):
        """Polishes the main window when it is first shown (before snapshots/self-checks)."""
        def __init__(self, controller):
            super().__init__()
            self.controller = controller

        def eventFilter(self, obj, event):
            if event.type() == QEvent.Type.Show and isinstance(obj, QMainWindow) and hasattr(obj, "predict_tab"):
                self.controller.attach(obj)
            return False

    class PredictionHero(QFrame):
        """Summary of the most recent prediction; mirrors _render_result's values and formats."""
        STATUS_KIND = {"PASS": "success", "WARN": "warning", "REVIEW": "warning", "BLOCKED": "danger"}
        PHASE_KIND = {"O3": "success", "P2": "danger", "BOUNDARY": "warning", "NOT_ASSESSABLE": "warning"}
        RISK_KIND = {"Low": "success", "Medium": "warning", "High": "danger"}

        def __init__(self, controller):
            super().__init__()
            self.controller = controller
            self.last = None
            self.setObjectName("heroCard")
            self.setMinimumWidth(300)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(18, 14, 18, 14)
            layout.setSpacing(2)
            top = QHBoxLayout()
            self.eyebrow = QLabel("LATEST PREDICTION")
            self.eyebrow.setObjectName("eyebrow")
            font = QFont(self.eyebrow.font())
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
            self.eyebrow.setFont(font)
            self.status = self._chip()
            top.addWidget(self.eyebrow)
            top.addStretch(1)
            top.addWidget(self.status)
            layout.addLayout(top)
            self.title = self._rich("heroTitle")
            self.title.setWordWrap(True)
            self.title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            layout.addWidget(self.title)
            layout.addSpacing(8)
            layout.addWidget(self._rich("statLabel", "Predicted " + MATH_HTML["DeltaV"]))
            self.value = self._rich("heroValue")
            layout.addWidget(self.value)
            self.caption = self._rich("heroCaption")
            self.caption.setWordWrap(True)
            layout.addWidget(self.caption)
            layout.addSpacing(6)
            chips = QHBoxLayout()
            chips.setSpacing(6)
            self.phase, self.risk = self._chip(), self._chip()
            chips.addWidget(self.phase)
            chips.addWidget(self.risk)
            chips.addStretch(1)
            layout.addLayout(chips)
            layout.addSpacing(10)
            grid = QGridLayout()
            grid.setHorizontalSpacing(22)
            grid.setVerticalSpacing(1)
            self.stats = {}
            for column, (key, unit) in enumerate((("Ravg", " (Å)"), ("PhiTM", " (Å<sup>−1</sup>)"), ("DeltaXNa", ""))):
                grid.addWidget(self._rich("statLabel", MATH_HTML[key] + unit), 0, column)
                self.stats[key] = QLabel("–")
                self.stats[key].setObjectName("statValue")
                grid.addWidget(self.stats[key], 1, column)
            grid.setColumnStretch(3, 1)
            layout.addLayout(grid)
            layout.addStretch(1)
            self.show_empty()

        @staticmethod
        def _rich(name, text=""):
            label = QLabel(text)
            label.setObjectName(name)
            label.setTextFormat(Qt.TextFormat.RichText)
            return label

        @staticmethod
        def _chip():
            label = QLabel()
            label.setProperty("chip", "neutral")
            label.hide()
            return label

        @staticmethod
        def _set_chip(label, text, kind):
            label.setText(text)
            label.setProperty("chip", kind)
            label.style().unpolish(label)
            label.style().polish(label)
            label.show()

        def _big(self, number, unit="", colour=None):
            t = self.controller.tokens
            html = "<span style='color:%s'>%s</span>" % (colour or t["text"], _ui_html.escape(number))
            if unit:
                html += "<span style='font-size:13pt; color:%s'>&nbsp;%s</span>" % (t["muted"], unit)
            return html

        def show_empty(self):
            self.last = None
            self.status.hide()
            self.phase.hide()
            self.risk.hide()
            self.title.setText("No prediction yet")
            self.value.setText(self._big("–", "", self.controller.tokens["faint"]))
            self.caption.setText("Run a prediction to see the signed " + MATH_HTML["DeltaV"] + " here.")
            for label in self.stats.values():
                label.setText("–")

        def show_result(self, title, result):
            self.last = (title, result)
            t = self.controller.tokens
            allowed = bool(result.get("prediction_allowed"))
            if allowed:
                status = "WARN" if result.get("warnings") else "PASS"
            else:
                status = "BLOCKED" if result.get("phase_code") == "P2" else "REVIEW"
            self._set_chip(self.status, status, self.STATUS_KIND.get(status, "neutral"))
            formula = str(result.get("formula", ""))
            self.title.setText("<b>%s</b>&nbsp;&nbsp;<span style='color:%s'>%s</span>"
                               % (_ui_html.escape(str(title)), t["muted"], _ui_formula_html(formula)))
            self.title.setToolTip(formula)
            if allowed:
                self.value.setText(self._big(format(float(result["DeltaV_pred"]), "+.2f"), "%"))
                self.caption.setText("CV-based empirical reference interval: [%s%%, %s%%]" % (
                    format(float(result["DeltaV_lower"]), ".2f"), format(float(result["DeltaV_upper"]), ".2f")))
                risk = str(result.get("risk", ""))
                self._set_chip(self.risk, "Risk: " + risk, self.RISK_KIND.get(risk, "neutral"))
            else:
                self.value.setText(self._big("Blocked", "", t["danger"]))
                self.caption.setText("Only O3-favored candidates pass the phase gate.")
                self.risk.hide()
            self._set_chip(self.phase, str(result.get("phase_label", "")),
                           self.PHASE_KIND.get(result.get("phase_code"), "neutral"))
            for key, spec in (("Ravg", ".4f"), ("PhiTM", ".4f"), ("DeltaXNa", ".2f")):
                value = result.get(key)
                self.stats[key].setText(format(float(value), spec) if value is not None else "–")

        def retheme(self):
            if self.last is None:
                self.show_empty()
            else:
                self.show_result(*self.last)

    return PolishFilter, PredictionHero


class _UIController:
    MODES = (("system", "Match system"), ("light", "Light"), ("dark", "Dark"))

    def __init__(self, app, base_app):
        self.app = app
        self.mode, self.persist = self._initial_mode()
        self.name = None
        self.tokens = _UI_THEMES["light"]
        self.icons = None
        self.window = None
        self.tracked = []
        self.refreshers = []
        self.applying = False
        self.PolishFilter, self.PredictionHero = _ui_widget_classes()
        self.font = _ui_pick_font()
        app.setStyle("Fusion")
        base_app.setFont(self.font)
        self.apply()
        try:
            app.styleHints().colorSchemeChanged.connect(self._system_scheme_changed)
        except Exception:
            pass
        self.filter = self.PolishFilter(self)
        app.installEventFilter(self.filter)

    @staticmethod
    def _initial_mode():
        forced = (_ui_os.environ.get(UI_THEME_ENV) or "").strip().lower()
        if forced in ("system", "light", "dark"):
            return forced, False
        try:
            from PySide6.QtCore import QSettings
            saved = str(QSettings("EM-VRF", "EM-VRF Desktop").value("ui/theme", "system")).lower()
            return (saved if saved in ("system", "light", "dark") else "system"), True
        except Exception:
            return "system", True

    def icon(self, key):
        from PySide6.QtGui import QIcon
        return QIcon(self.icons + "/" + key + ".svg") if self.icons else QIcon()

    def apply(self):
        if self.applying:
            return
        self.applying = True
        try:
            from PySide6.QtCore import Qt
            hints = self.app.styleHints()
            scheme = {"light": Qt.ColorScheme.Light, "dark": Qt.ColorScheme.Dark}.get(self.mode, Qt.ColorScheme.Unknown)
            try:
                hints.setColorScheme(scheme)  # keeps the native title bar in step with the theme
            except Exception:
                pass
            if self.mode in ("light", "dark"):
                self.name = self.mode
            else:
                self.name = "dark" if hints.colorScheme() == Qt.ColorScheme.Dark else "light"
            self.tokens = _UI_THEMES[self.name]
            self.icons = _ui_write_icons(self.name, self.tokens)
            self.app.setPalette(_ui_palette(self.tokens))
            self.app.setStyleSheet(_ui_stylesheet(self.tokens, self.icons))
            for widget in list(self.tracked):
                try:
                    widget.setStyleSheet(widget._ui_legacy_css)
                except Exception:
                    self.tracked.remove(widget)
            for refresh in self.refreshers:
                try:
                    refresh()
                except Exception:
                    _ui_report("theme refresh")
        finally:
            self.applying = False

    def _system_scheme_changed(self, *args):
        if self.mode == "system" and not self.applying:
            self.apply()

    def set_mode(self, mode):
        self.mode = mode
        if self.persist:
            try:
                from PySide6.QtCore import QSettings
                QSettings("EM-VRF", "EM-VRF Desktop").setValue("ui/theme", mode)
            except Exception:
                pass
        self.apply()

    def track(self, widget):
        """Route the widget's inline styles through the theme (the original code keeps setting them)."""
        if hasattr(widget, "_ui_legacy_css"):
            return
        native = widget.setStyleSheet
        controller = self

        def themed_set_style_sheet(css, _native=native, _widget=widget):
            _widget._ui_legacy_css = css
            _native(_ui_translate_css(css, controller.tokens))

        widget._ui_legacy_css = widget.styleSheet()
        widget.setStyleSheet = themed_set_style_sheet
        self.tracked.append(widget)
        native(_ui_translate_css(widget._ui_legacy_css, self.tokens))

    def attach(self, window):
        if self.window is not None:
            return
        self.window = window
        try:
            self.app.removeEventFilter(self.filter)
        except Exception:
            pass
        for name, step in (("header", self._polish_header), ("prediction page", self._polish_predict),
                           ("model information", self._polish_model_info)):
            try:
                step(window)
            except Exception:
                _ui_report(name)
        # Object names and variant properties were set after the first polish; re-polish everything once.
        self.apply()

    def _polish_header(self, window):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QActionGroup
        from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QMenu, QSizePolicy, QSpacerItem, QToolButton, QWidget
        tabs = window.tabs
        tabs.setObjectName("mainTabs")
        tabs.setDocumentMode(True)
        bar = tabs.tabBar()
        bar.setDrawBase(False)
        bar.setExpanding(False)
        bar.setCursor(Qt.CursorShape.PointingHandCursor)

        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(18, 0, 14, 0)
        brand_layout.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(window.windowIcon().pixmap(26, 26))
        name = QLabel("EM-VRF Desktop")
        name.setObjectName("brandName")
        divider = QFrame()
        divider.setObjectName("brandDivider")
        divider.setFixedSize(1, 22)
        # Qt bottom-aligns corner widgets; a tall spacer makes them fill the tab row so content centres.
        row_filler = (0, 200, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        centre = Qt.AlignmentFlag.AlignVCenter
        brand_layout.addSpacerItem(QSpacerItem(*row_filler))
        brand_layout.addWidget(logo, 0, centre)
        brand_layout.addWidget(name, 0, centre)
        brand_layout.addSpacing(8)
        brand_layout.addWidget(divider, 0, centre)
        tabs.setCornerWidget(brand, Qt.Corner.TopLeftCorner)
        brand.show()  # the window is already visible, so new children start hidden

        actions = QWidget()
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 16, 0)
        actions_layout.setSpacing(10)
        version = QLabel(str(APP_VERSION))
        version.setObjectName("versionChip")
        version.setToolTip("Application version " + str(APP_VERSION) + "\nUI revision " + UI_REVISION)
        button = QToolButton()
        button.setObjectName("themeButton")
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip("Appearance")
        menu = QMenu(button)
        group = QActionGroup(menu)
        group.setExclusive(True)
        mode_actions = {}
        for key, label in self.MODES:
            action = menu.addAction(label)
            action.setCheckable(True)
            group.addAction(action)
            action.triggered.connect(lambda checked=False, key=key: self.set_mode(key))
            mode_actions[key] = action
        button.setMenu(menu)
        actions_layout.addSpacerItem(QSpacerItem(*row_filler))
        actions_layout.addWidget(version, 0, centre)
        actions_layout.addWidget(button, 0, centre)
        tabs.setCornerWidget(actions, Qt.Corner.TopRightCorner)
        actions.show()

        def refresh_header():
            button.setText(dict(self.MODES)[self.mode].replace("Match system", "System"))
            button.setIcon(self.icon("theme_" + (self.mode if self.mode != "system" else "system")))
            mode_actions[self.mode].setChecked(True)
        self.refreshers.append(refresh_header)
        refresh_header()

    def _polish_predict(self, window):
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                                       QVBoxLayout, QWidget)
        tab = window.predict_tab
        tab.layout().setContentsMargins(16, 4, 16, 16)
        tab.layout().setSpacing(16)
        pointer = Qt.CursorShape.PointingHandCursor

        def variant(button, kind):
            button.setProperty("variant", kind)
            button.setCursor(pointer)

        # Mode selection page: title + two descriptive cards.
        select_layout = tab.page_select.layout()
        select_layout.setContentsMargins(8, 0, 8, 0)
        title = next((w for w in tab.page_select.findChildren(QLabel) if w.text() == "Select Prediction Mode"), None)
        if title is not None:
            title.setStyleSheet("")
            title.setObjectName("pageTitle")
            subtitle = QLabel("Predict the signed unit-cell volume change Δ<i>V</i> of O3-type layered "
                              "sodium-ion cathodes.")
            subtitle.setObjectName("pageSubtitle")
            subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            subtitle.setWordWrap(True)
            select_layout.insertWidget(select_layout.indexOf(title) + 1, subtitle)
            select_layout.insertSpacing(select_layout.indexOf(subtitle) + 1, 18)
        mode_icons = []
        for button, icon, heading, text in (
                (tab.btn_go_formula, "mode_formula", "Formula Input",
                 "Type a chemical formula. Charge, TM_sum and the phase gate update as you type; "
                 "batch CSV/Excel prediction is on the same page."),
                (tab.btn_go_designer, "mode_designer", "Element Designer",
                 "Pick dopants from the periodic selector and set each stoichiometric coefficient.")):
            button.setStyleSheet("")
            button.setText("")
            button.setAccessibleName(heading)
            button.setFixedHeight(100)
            variant(button, "card")
            row = QHBoxLayout(button)
            row.setContentsMargins(18, 14, 16, 14)
            row.setSpacing(16)
            badge = QLabel()
            badge.setObjectName("modeCardIcon")
            badge.setFixedSize(48, 48)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column = QVBoxLayout()
            column.setSpacing(3)
            head = QLabel(heading)
            head.setObjectName("modeCardTitle")
            body = QLabel(text.replace("TM_sum", MATH_HTML["TMSum"]))
            body.setObjectName("modeCardText")
            body.setTextFormat(Qt.TextFormat.RichText)
            body.setWordWrap(True)
            column.addStretch(1)
            column.addWidget(head)
            column.addWidget(body)
            column.addStretch(1)
            arrow = QLabel("›")
            arrow.setObjectName("modeCardArrow")
            for child in (badge, head, body, arrow):
                child.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            row.addWidget(badge)
            row.addLayout(column, 1)
            row.addWidget(arrow)
            mode_icons.append((badge, icon))

        def refresh_mode_icons():
            for badge, icon in mode_icons:
                badge.setPixmap(self.icon(icon).pixmap(26, 26))
        self.refreshers.append(refresh_mode_icons)
        refresh_mode_icons()

        # Back buttons become compact ghost links.
        for page in (tab.page_formula, tab.page_designer):
            for button in page.findChildren(QPushButton):
                if button.text() == "← Back":
                    button.setText("←  Prediction modes")
                    variant(button, "ghost")
                    button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
                    page.layout().setAlignment(button, Qt.AlignmentFlag.AlignLeft)
            page.layout().setContentsMargins(0, 0, 4, 0)
            page.layout().setSpacing(12)

        for button in (tab.btn_formula, tab.btn_designer):
            variant(button, "primary")
            button.setMinimumHeight(40)
        for button in (tab.btn_batch_load, tab.btn_batch_export):
            button.setCursor(pointer)
        variant(tab.btn_clear_comparison, "ghost")

        for form in tab.findChildren(QFormLayout):
            form.setHorizontalSpacing(16)
            form.setVerticalSpacing(10)
            form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        for box in tab.findChildren(QGroupBox):
            if box.layout() is not None:
                box.layout().setContentsMargins(4, 4, 4, 4)
                box.layout().setSpacing(10)

        # Periodic selector: element tiles instead of tick boxes.
        tab.selector.setObjectName("periodicSelector")
        tab.selector.layout().setContentsMargins(0, 0, 0, 0)
        for box in tab.selector.findChildren(QGroupBox):
            box.setObjectName("subGroup")
            grid = box.layout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(4)
            grid.setVerticalSpacing(4)
            grid.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        from PySide6.QtGui import QFontMetrics
        tile_font = tab.selector.font()
        tile_font.setWeight(tile_font.Weight.DemiBold)
        tile_metrics = QFontMetrics(tile_font)
        tile_width, tile_height = 32, 30
        for symbol, check in tab.selector.checkboxes.items():
            check.setCursor(pointer)
            check.setFixedSize(tile_width, tile_height)
            # QCheckBox has no text-align; centre the symbol with a computed left padding.
            check.setStyleSheet("QCheckBox { padding-left: %dpx; }"
                                % max(0, (tile_width - 2 - tile_metrics.horizontalAdvance(symbol)) // 2))

        # Inline label styles (hints, badges, guards) follow the theme.
        for label in (tab.formula_charge_lbl, tab.formula_tmsum_lbl, tab.formula_phase_lbl,
                      tab.designer_charge_lbl, tab.designer_tmsum_lbl, tab.designer_phase_lbl):
            label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        for label in (tab.formula_link_hint, tab.designer_link_hint):
            label.setWordWrap(True)
        for label in window.findChildren(QLabel):
            if label.styleSheet():
                if "font-size: 11px" in label.styleSheet():
                    label.setWordWrap(True)
                self.track(label)
        for label in (tab.formula_charge_lbl, tab.formula_tmsum_lbl, tab.formula_phase_lbl,
                      tab.designer_charge_lbl, tab.designer_tmsum_lbl, tab.designer_phase_lbl,
                      tab.formula_link_hint, tab.designer_link_hint):
            self.track(label)

        # Batch file label: file name, full path in the tooltip.
        file_label = tab.batch_lbl
        file_label.setObjectName("fileLabel")
        file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        native_set_text = file_label.setText

        def set_file_text(text, _native=native_set_text, _label=file_label):
            text = str(text)
            if "/" in text or "\\" in text:
                _label.setToolTip(text)
                _native(_UIPath(text).name)
            else:
                _label.setToolTip("")
                _native(text)
        file_label.setText = set_file_text
        file_label.setText(file_label.text())

        for table in (tab.batch_table, tab.comparison_table):
            table.setShowGrid(False)
            table.setAlternatingRowColors(True)
            table.setFrameShape(QFrame.Shape.NoFrame)
            table.verticalHeader().setDefaultSectionSize(34)
            table.horizontalHeader().setHighlightSections(False)
            table.verticalHeader().setHighlightSections(False)

        chart_scroll = tab.comparison_chart.parentWidget().parentWidget()
        chart_scroll.setObjectName("chartScroll")
        chart_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._comparison_switch(tab, chart_scroll)
        type(tab.comparison_chart).paintEvent = _ui_chart_paint
        self.refreshers.append(tab.comparison_chart.update)

        # Results column: latest-prediction card beside the log, then the comparison card.
        results_layout = tab.output.parentWidget().layout()
        results_layout.setSpacing(16)
        results_layout.removeWidget(tab.output)
        hero = self.PredictionHero(self)
        log_card = QFrame()
        log_card.setProperty("card", True)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 12, 6, 6)
        log_layout.setSpacing(4)
        log_header = QHBoxLayout()
        log_title = QLabel("Prediction log")
        log_title.setObjectName("cardTitle")
        clear_log = QPushButton("Clear")
        variant(clear_log, "ghost")
        clear_log.setToolTip("Clear the prediction log (the comparison is kept)")
        clear_log.clicked.connect(tab.output.clear)
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        log_header.addWidget(clear_log)
        log_header.setContentsMargins(0, 0, 10, 0)
        log_layout.addLayout(log_header)
        tab.output.setMaximumHeight(16777215)  # fill the card; the row height follows the summary card
        log_layout.addWidget(tab.output, 1)
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(16)
        top_layout.addWidget(hero, 2)
        top_layout.addWidget(log_card, 3)
        results_layout.insertWidget(0, top_row)
        tab.ui_prediction_card = hero
        self.refreshers.append(hero.retheme)

        native_render = tab._render_result

        def render_result(title, result, _native=native_render):
            outcome = _native(title, result)
            try:
                hero.show_result(title, result)
            except Exception:
                _ui_report("prediction card")
            return outcome
        tab._render_result = render_result

    def _comparison_switch(self, tab, chart_scroll):
        """Show the comparison table and chart one at a time, each at full card width (no splitter)."""
        from PySide6.QtCore import QSettings, Qt
        from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QPushButton, QStackedWidget
        splitter = tab.comparison_table.parentWidget()
        card_layout = splitter.parentWidget().layout()
        views = QStackedWidget()
        card_layout.insertWidget(card_layout.indexOf(splitter), views, 1)
        views.addWidget(tab.comparison_table)  # reparenting takes both views out of the splitter
        views.addWidget(chart_scroll)
        card_layout.removeWidget(splitter)
        splitter.hide()
        splitter.deleteLater()
        tab.comparison_table.horizontalHeader().setStretchLastSection(True)

        toolbar = next(item.layout() for item in (card_layout.itemAt(i) for i in range(card_layout.count()))
                       if item.layout() is not None and item.layout().indexOf(tab.btn_clear_comparison) >= 0)
        track = QFrame()
        track.setObjectName("segmented")
        row = QHBoxLayout(track)
        row.setContentsMargins(3, 3, 3, 3)
        row.setSpacing(2)
        group = QButtonGroup(track)
        buttons = {}
        for key, label in (("table", "Table"), ("chart", "Chart")):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            group.addButton(button)
            row.addWidget(button)
            buttons[key] = button
        toolbar.insertWidget(toolbar.indexOf(tab.btn_clear_comparison), track)
        toolbar.insertSpacing(toolbar.indexOf(tab.btn_clear_comparison), 6)

        def show(key, remember=True):
            views.setCurrentWidget(tab.comparison_table if key == "table" else chart_scroll)
            buttons[key].setChecked(True)
            if remember and self.persist:
                try:
                    QSettings("EM-VRF", "EM-VRF Desktop").setValue("ui/comparison_view", key)
                except Exception:
                    pass

        for key, button in buttons.items():
            button.clicked.connect(lambda checked=False, key=key: show(key))
        initial = "table"
        if self.persist:
            try:
                saved = str(QSettings("EM-VRF", "EM-VRF Desktop").value("ui/comparison_view", "table"))
                initial = saved if saved in buttons else "table"
            except Exception:
                pass
        show(initial, remember=False)
        tab.ui_show_comparison = show

    def _polish_model_info(self, window):
        from PySide6.QtWidgets import QHBoxLayout
        info = window.model_info_tab
        layout = info.layout()
        layout.setContentsMargins(24, 8, 24, 24)
        info.text.setObjectName("docView")
        info.text.setMaximumWidth(1040)
        info.text.document().setDocumentMargin(28)
        # Centre a readable-width document column; stretches take the space beyond the maximum width.
        layout.removeWidget(info.text)
        column = QHBoxLayout()
        column.addStretch(1)
        column.addWidget(info.text, 100)
        column.addStretch(1)
        layout.addLayout(column)
        native_set_html = info.text.setHtml
        controller = self

        def themed_set_html(markup, _native=native_set_html):
            css = _ui_doc_css(controller.tokens, controller.font.family())
            _native(_ui_re.sub(r"<style>.*?</style>", lambda match: css, str(markup), count=1, flags=_ui_re.S))
        info.text.setHtml = themed_set_html
        self.refreshers.append(info.refresh)
        info.refresh()


def _ui_application_class(base):
    class ModernApplication(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ui_controller = None
            try:
                self._ui_controller = _UIController(self, base)
                _UI["controller"] = self._ui_controller
            except Exception:
                _ui_report("theme setup")

        def setFont(self, *args, **kwargs):
            # run_gui() asks for Arial 10; keep the themed UI font instead.
            controller = getattr(self, "_ui_controller", None)
            if controller is not None:
                return base.setFont(controller.font)
            return base.setFont(*args, **kwargs)
    return ModernApplication


def run_gui(smoke_test=False, snapshot_path=None):
    from PySide6 import QtWidgets
    previous = QtWidgets.QApplication
    QtWidgets.QApplication = _ui_application_class(previous)
    try:
        return _precision_run_gui(smoke_test=smoke_test, snapshot_path=snapshot_path)
    finally:
        QtWidgets.QApplication = previous


if __name__ == "__main__":
    raise SystemExit(main())
