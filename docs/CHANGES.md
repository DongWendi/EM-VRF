# Revision history

The fitted model (`2026.9.14-unified-45-literature-only`) is unchanged in all revisions below. The application reports version `2026.10.03-full-precision-r3`; later revisions change the interface, numeric input stepping and packaging only.

## 2026.10.09 — source release

- `main.py` is the complete application source in one file; the released `EM-VRF Desktop.exe` embeds exactly this file (check with `tools/build_exe.py verify`).
- ΔV Comparison card: a Table / Chart switch replaces the draggable divider, so each view uses the full card width. The last choice is remembered.
- `train_model.py` reproduces the shipped model and its cross-validation metrics from Supplementary Table 5.
- MIT license.
- `data/training_population.json`: removed a provenance statement that referred to a document not distributed with the software.
- Excel document properties of `Supplementary Table 5.xlsx` and `Example.xlsx` cleaned (author names, local save path, localized font and style names); cell data unchanged.

## 2026.10.08 — interface

- Light and dark themes following Windows (System / Light / Dark switch), card layout, latest-prediction summary card, periodic-table dopant picker, comparison chart redrawn with theme colours and rounded tick values.
- Fix: mouse-wheel, arrow and PageUp/PageDown steps in numeric fields are taken on the decimal grid (0.09 + 0.01 = 0.1, not 0.09999999999999999); typed full-precision values are kept.

## 2026.10.03 — full-precision revision (r3)

- Model inputs keep full floating-point precision (no two-decimal rounding); two-decimal display only.
- Ce⁴⁺ (CN = VI) effective ionic radius corrected from 0.530 to 0.870 Å (Shannon 1976, Table 1, DOI 10.1107/S0567739476001551); none of the 45 training formulas contains Ce.
- Records: `precision_revision/`.
