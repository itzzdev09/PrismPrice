"""Build and load a small, time-validated demand model bundle for local serving."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from prismprice.estimation.demand import (
    HistGradientBoostingDemandModel,
    QuantileDemandModel,
    compare_demand_models,
)
from prismprice.features.builder import FeatureBuilder

__all__ = [
    "DemandBundleError",
    "build_demand_bundle",
    "load_demand_bundle",
]

_MODEL_FILENAME = "demand_model.joblib"
_MANIFEST_FILENAME = "demand_manifest.json"
_DATA_COLUMNS = ["sku", "date", "units", "list_price"]
_MIN_HISTORY_DAYS = 91


class DemandBundleError(RuntimeError):
    """The demand bundle is absent, inconsistent, or failed its release gate."""


def _panel_snapshot_id(panel: pd.DataFrame) -> str:
    stable = panel[_DATA_COLUMNS].sort_values(["sku", "date"]).reset_index(drop=True)
    hashed_rows = pd.util.hash_pandas_object(stable, index=False, categorize=True)
    return f"sha256:{hashlib.sha256(hashed_rows.values.tobytes()).hexdigest()}"


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def build_demand_bundle(
    panel: pd.DataFrame,
    output_dir: str | Path,
    *,
    min_observations: int = 300,
    stride: int = 4,
    n_folds: int = 3,
    seed: int = 20260817,
) -> dict[str, Any]:
    """Compare candidate quantile models on temporal folds, then save the winner.

    The input panel contains active-sales days only. The returned bundle records
    that unit explicitly and is not suitable for retailers that need latent
    demand correction for stockouts or unobserved availability.
    """
    required = {"sku", "date", "units", "list_price"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Demand panel is missing columns: {sorted(missing)}")
    if min_observations < _MIN_HISTORY_DAYS + 40:
        raise ValueError("min_observations must leave at least 40 training rows after feature warmup")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    data = panel[_DATA_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.dropna(subset=["sku", "date", "units", "list_price"])
    data = data[(data["units"] > 0) & (data["list_price"] > 0)]
    data = data.sort_values(["sku", "date"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("Demand panel contains no positive active-sales observations")

    # The source has both weighted realised price and the daily list price.
    # Demand response uses list price, which is the retailer-controlled price.
    feature_panel = data[["sku", "date", "units", "list_price"]].rename(
        columns={"list_price": "price"}
    )
    builder = FeatureBuilder(feature_panel)
    training_parts = [
        builder.training_frame(sku, min_history_days=_MIN_HISTORY_DAYS, stride=stride)
        for sku, group in feature_panel.groupby("sku", sort=True)
        if len(group) >= min_observations
    ]
    training_parts = [part for part in training_parts if not part.empty]
    if not training_parts:
        raise DemandBundleError(
            f"No SKU has at least {min_observations} active-sales days."
        )
    training = pd.concat(training_parts).sort_index(level="as_of")
    if len(training) < 200 or training.index.get_level_values("as_of").nunique() < n_folds + 1:
        raise DemandBundleError("The selected historical panel is too small for temporal validation.")

    def lightgbm_factory() -> QuantileDemandModel:
        return QuantileDemandModel(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=30,
            conformal_fraction=0.25,
            seed=seed,
        )

    def sklearn_factory() -> HistGradientBoostingDemandModel:
        return HistGradientBoostingDemandModel(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=30,
            conformal_fraction=0.25,
            seed=seed,
        )

    comparison = compare_demand_models(
        training,
        model_factories={
            "lightgbm_quantile": lightgbm_factory,
            "sklearn_hist_gradient_boosting_quantile": sklearn_factory,
        },
        n_folds=n_folds,
        interval_tolerance_pp=5.0,
        tail_tolerance_pp=5.0,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    comparison_payload = comparison.as_dict()
    _write_json_atomic(destination / "demand_comparison.json", comparison_payload)
    if comparison.selected_model is None:
        raise DemandBundleError(
            "No candidate passed the rolling-origin calibration gates. "
            f"Review {destination / 'demand_comparison.json'}; no serving bundle was published."
        )

    model_factories = {
        "lightgbm_quantile": lightgbm_factory,
        "sklearn_hist_gradient_boosting_quantile": sklearn_factory,
    }
    model = model_factories[comparison.selected_model]().fit(training)
    scorecard = comparison.candidates[comparison.selected_model]
    training_through = pd.Timestamp(data["date"].max()).normalize()
    decision_as_of = training_through + pd.Timedelta(days=1)
    snapshot_id = _panel_snapshot_id(data)
    supported_skus = sorted(training.index.get_level_values("sku").astype(str).unique())

    import joblib

    model_path = destination / _MODEL_FILENAME
    manifest_path = destination / _MANIFEST_FILENAME
    temporary_model: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination, suffix=".joblib.tmp", delete=False
        ) as handle:
            temporary_model = Path(handle.name)
        joblib.dump(model, temporary_model, compress=3)
        artifact_hash = hashlib.sha256(temporary_model.read_bytes()).hexdigest()
        model_version = f"{comparison.selected_model}-{artifact_hash[:16]}"
        manifest = {
            "format_version": 1,
            "model_name": comparison.selected_model,
            "model_version": model_version,
            "artifact_sha256": artifact_hash,
            "dataset_name": "UCI Online Retail II",
            "dataset_snapshot_id": snapshot_id,
            "demand_unit": "units per active sales day",
            "training_from": pd.Timestamp(data["date"].min()).isoformat(),
            "training_through": training_through.isoformat(),
            "decision_as_of": decision_as_of.isoformat(),
            "supported_skus": supported_skus,
            "training_rows": len(training),
            "validation": {
                "method": "expanding-window rolling-origin backtest",
                "n_folds": len(scorecard.folds),
                **asdict(scorecard.pooled),
                "calibrated": scorecard.calibrated,
                "folds": [asdict(report) for report in scorecard.folds],
                "comparison": comparison_payload,
                "release_gate": {
                    "interval_coverage_tolerance_pp": 5.0,
                    "tail_coverage_tolerance_pp": 5.0,
                },
            },
            "feature_names": list(model.feature_names),
            "assumptions": [
                "Training uses positive sales on active SKU-days; unavailable dates are not imputed as zero demand.",
                "The source has no stockout or availability labels, so the model is not an uncensored latent-demand estimate.",
                "The source has no unit costs; the workbench cost field is an operator-supplied assumption.",
                "Competitor prices, inventory levels, retention value, and cross-SKU substitution are not model inputs in this bundle.",
                "The model is fitted on historical data through 2011-12-09 and is a research demonstration, not a current market recommendation.",
            ],
        }
        os.replace(temporary_model, model_path)
        temporary_model = None
        _write_json_atomic(manifest_path, manifest)
    finally:
        if temporary_model is not None and temporary_model.exists():
            temporary_model.unlink()

    return manifest


def load_demand_bundle(bundle_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load only a complete, calibration-gated local bundle with a matching digest."""
    root = Path(bundle_dir)
    model_path = root / _MODEL_FILENAME
    manifest_path = root / _MANIFEST_FILENAME
    if not model_path.is_file() or not manifest_path.is_file():
        raise DemandBundleError(
            f"No serving bundle at {root}; build it with scripts/build_demand_bundle.py."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if artifact_hash != manifest.get("artifact_sha256"):
        raise DemandBundleError("Demand model artifact digest does not match its manifest.")
    if manifest.get("format_version") != 1:
        raise DemandBundleError("Unsupported demand bundle format version.")
    if not manifest.get("validation", {}).get("calibrated", False):
        raise DemandBundleError("Demand model has not passed its rolling-origin calibration gate.")

    import joblib

    model = joblib.load(model_path)
    if tuple(getattr(model, "feature_names", ())) != tuple(manifest.get("feature_names", ())):
        raise DemandBundleError("Demand model feature schema does not match its manifest.")
    return model, manifest
