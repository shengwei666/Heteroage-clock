"""
heteroage_clock.core.metrics

Industrial-grade metric utilities used consistently across Stage 1/2/3.

Design goals:
- Deterministic, well-defined behavior on missing values (NaNs).
- Support both micro metrics (sample-level) and macro metrics (group-averaged, e.g., Tissue).
- Provide a single canonical "score" definition compatible with the original research scripts:
    Score = MicroMAE + lambda * MacroMAE
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score


ArrayLike = Union[np.ndarray, Sequence[float], pd.Series]


@dataclass(frozen=True)
class RegressionMetrics:
    """Container for standard regression metrics."""
    n_total: int
    n_valid: int
    micro_mae: float
    median_ae: float
    r2: float
    macro_mae: Optional[float] = None
    per_group_mae: Optional[Dict[str, float]] = None


def _to_1d_float_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input into a 1D float numpy array."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after reshape; got shape {arr.shape}")
    return arr


def _validate_lengths(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"Length mismatch: y_true={y_true.shape[0]}, y_pred={y_pred.shape[0]}")


def _valid_mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Valid samples are those where both true and pred are finite."""
    return np.isfinite(y_true) & np.isfinite(y_pred)


def compute_micro_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[int, int, float, float, float]:
    """
    Compute micro (sample-level) metrics with robust NaN/inf handling.

    Returns:
        n_total, n_valid, micro_mae, median_ae, r2
    """
    yt = _to_1d_float_array(y_true, "y_true")
    yp = _to_1d_float_array(y_pred, "y_pred")
    _validate_lengths(yt, yp)

    n_total = int(yt.shape[0])
    mask = _valid_mask(yt, yp)
    n_valid = int(mask.sum())

    if n_valid == 0:
        return n_total, 0, float("nan"), float("nan"), float("nan")

    yt_v = yt[mask]
    yp_v = yp[mask]

    micro = float(mean_absolute_error(yt_v, yp_v))
    medae = float(median_absolute_error(yt_v, yp_v))

    # r2 is undefined for n_valid < 2 or constant y; sklearn returns 0.0 in some edge cases.
    # We keep sklearn behavior but still guard for pathological sizes.
    if n_valid < 2:
        r2 = float("nan")
    else:
        try:
            r2 = float(r2_score(yt_v, yp_v))
        except Exception:
            r2 = float("nan")

    return n_total, n_valid, micro, medae, r2


def compute_macro_mae_by_group(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: Sequence[Any],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute MacroMAE as the mean of per-group MAEs.

    This matches the Stage 0/1 approach in your scripts:
        df.groupby('tissue').apply(lambda x: MAE(x['age'], x['pred'])).mean()

    Notes:
    - Samples with non-finite y_true/y_pred are excluded *within* each group.
    - Groups are stringified in output for stable JSON/CSV export.
    - Groups with <1 valid sample are skipped.
    """
    yt = _to_1d_float_array(y_true, "y_true")
    yp = _to_1d_float_array(y_pred, "y_pred")
    _validate_lengths(yt, yp)

    if len(groups) != yt.shape[0]:
        raise ValueError(f"Length mismatch: groups={len(groups)}, y={yt.shape[0]}")

    df = pd.DataFrame(
        {
            "group": pd.Series(groups, dtype="object"),
            "y_true": yt,
            "y_pred": yp,
        }
    )

    per_group: Dict[str, float] = {}
    for g, sub in df.groupby("group", dropna=False):
        m = np.isfinite(sub["y_true"].to_numpy()) & np.isfinite(sub["y_pred"].to_numpy())
        if int(m.sum()) == 0:
            continue
        mae = float(mean_absolute_error(sub.loc[m, "y_true"], sub.loc[m, "y_pred"]))
        per_group[str(g)] = mae

    if not per_group:
        return float("nan"), {}

    macro = float(np.mean(list(per_group.values())))
    return macro, per_group


def compute_regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    groups: Optional[Sequence[Any]] = None,
) -> RegressionMetrics:
    """
    Compute a consistent set of metrics used across the pipeline.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        groups: Optional grouping labels (e.g., Tissue) for MacroMAE.

    Returns:
        RegressionMetrics dataclass with micro + (optional) macro metrics.
    """
    n_total, n_valid, micro, medae, r2 = compute_micro_metrics(y_true, y_pred)

    if groups is None:
        return RegressionMetrics(
            n_total=n_total,
            n_valid=n_valid,
            micro_mae=micro,
            median_ae=medae,
            r2=r2,
            macro_mae=None,
            per_group_mae=None,
        )

    macro, per_group = compute_macro_mae_by_group(y_true, y_pred, groups)
    return RegressionMetrics(
        n_total=n_total,
        n_valid=n_valid,
        micro_mae=micro,
        median_ae=medae,
        r2=r2,
        macro_mae=macro,
        per_group_mae=per_group,
    )


def heteroage_score(micro_mae: float, macro_mae: float, lam: float = 1.0) -> float:
    """
    Canonical scoring function used for sweep selection in Stage 1:
        Score = MicroMAE + lambda * MacroMAE

    Args:
        micro_mae: sample-level MAE
        macro_mae: group-averaged MAE (e.g., averaged across tissues)
        lam: weight on macro term

    Returns:
        score (float)
    """
    if not np.isfinite(micro_mae) or not np.isfinite(macro_mae):
        return float("nan")
    return float(micro_mae + float(lam) * macro_mae)
