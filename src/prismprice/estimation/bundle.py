"""Build and load a small, time-validated demand model bundle for local serving."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
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
    "LoadedDemandBundle",
    "build_demand_bundle",
    "load_demand_bundle",
    "load_demand_bundle_with_backup",
]

_MODEL_FILENAME = "demand_model.joblib"
_BACKUP_MODEL_FILENAME = "demand_backup_model.joblib"
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
    eligible = sorted(
        (score for score in comparison.candidates.values() if score.calibrated),
        key=lambda score: score.pooled.pinball_loss,
    )
    primary_score = comparison.candidates[comparison.selected_model]
    backup_score = next(
        (score for score in eligible if score.model_name != comparison.selected_model),
        None,
    )
    model = model_factories[comparison.selected_model]().fit(training)
    backup_model = (
        model_factories[backup_score.model_name]().fit(training)
        if backup_score is not None
        else None
    )
    training_through = pd.Timestamp(data["date"].max()).normalize()
    decision_as_of = training_through + pd.Timedelta(days=1)
    snapshot_id = _panel_snapshot_id(data)
    supported_skus = sorted(training.index.get_level_values("sku").astype(str).unique())

    import joblib

    model_path = destination / _MODEL_FILENAME
    backup_model_path = destination / _BACKUP_MODEL_FILENAME
    manifest_path = destination / _MANIFEST_FILENAME
    temporary_models: list[tuple[Path, Path]] = []
    try:
        def write_model_temp(fitted_model: Any, final_path: Path) -> tuple[Path, str]:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination, suffix=".joblib.tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
            temporary_models.append((temporary_path, final_path))
            joblib.dump(fitted_model, temporary_path, compress=3)
            return temporary_path, hashlib.sha256(temporary_path.read_bytes()).hexdigest()

        temporary_model, artifact_hash = write_model_temp(model, model_path)
        model_version = f"{comparison.selected_model}-{artifact_hash[:16]}"
        backup_artifact: dict[str, Any] | None = None
        if backup_model is not None and backup_score is not None:
            _, backup_hash = write_model_temp(backup_model, backup_model_path)
            backup_artifact = {
                "filename": _BACKUP_MODEL_FILENAME,
                "model_name": backup_score.model_name,
                "model_version": f"{backup_score.model_name}-{backup_hash[:16]}",
                "artifact_sha256": backup_hash,
                "feature_names": list(backup_model.feature_names),
                "dataset_snapshot_id": snapshot_id,
                "decision_as_of": decision_as_of.isoformat(),
                "supported_skus": supported_skus,
                "validation": {
                    "method": "expanding-window rolling-origin backtest",
                    "n_folds": len(backup_score.folds),
                    **asdict(backup_score.pooled),
                    "calibrated": backup_score.calibrated,
                    "folds": [asdict(report) for report in backup_score.folds],
                    "comparison": comparison_payload,
                },
            }
        manifest = {
            "format_version": 2,
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
                "n_folds": len(primary_score.folds),
                **asdict(primary_score.pooled),
                "calibrated": primary_score.calibrated,
                "folds": [asdict(report) for report in primary_score.folds],
                "comparison": comparison_payload,
                "release_gate": {
                    "interval_coverage_tolerance_pp": 5.0,
                    "tail_coverage_tolerance_pp": 5.0,
                },
            },
            "backup_model": backup_artifact,
            "backup_model_status": (
                "validated" if backup_artifact is not None
                else "unavailable_no_other_candidate_passed_calibration"
            ),
            "feature_names": list(model.feature_names),
            "assumptions": [
                "Training uses positive sales on active SKU-days; unavailable dates are not imputed as zero demand.",
                "The source has no stockout or availability labels, so the model is not an uncensored latent-demand estimate.",
                "The source has no unit costs; the workbench cost field is an operator-supplied assumption.",
                "Competitor prices, inventory levels, retention value, and cross-SKU substitution are not model inputs in this bundle.",
                "The model is fitted on historical data through 2011-12-09 and is a research demonstration, not a current market recommendation.",
            ],
        }
        for temporary_path, final_path in temporary_models:
            os.replace(temporary_path, final_path)
        _write_json_atomic(manifest_path, manifest)
    finally:
        for temporary_path, _ in temporary_models:
            if temporary_path.exists():
                temporary_path.unlink()

    return manifest


def load_demand_bundle(bundle_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a verified bundle, retaining the historical two-value API."""
    loaded = load_demand_bundle_with_backup(bundle_dir)
    return loaded.model, loaded.active_manifest


@dataclass(frozen=True)
class LoadedDemandBundle:
    """Verified active model and optional standby model, with load diagnostics."""

    model: Any
    active_manifest: dict[str, Any]
    backup_model: Any | None
    backup_manifest: dict[str, Any] | None
    primary_error: str | None = None
    backup_error: str | None = None


def _load_model_artifact(
    root: Path, filename: str, digest: str, feature_names: list[str]
) -> Any:
    model_path = root / filename
    try:
        if not model_path.is_file():
            raise DemandBundleError(f"Demand model artifact {filename!r} is missing.")
        artifact_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DemandBundleError(
            f"Demand model artifact {filename!r} could not be read: {exc}"
        ) from exc
    if artifact_hash != digest:
        raise DemandBundleError(
            f"Demand model artifact digest does not match its manifest: {filename}."
        )

    try:
        import joblib

        model = joblib.load(model_path)
    except Exception as exc:
        raise DemandBundleError(
            f"Demand model artifact {filename!r} could not be loaded: {exc}"
        ) from exc
    if tuple(getattr(model, "feature_names", ())) != tuple(feature_names):
        raise DemandBundleError(
            f"Demand model feature schema does not match its manifest: {filename}."
        )
    return model


def load_demand_bundle_with_backup(bundle_dir: str | Path) -> LoadedDemandBundle:
    """Load each calibration-gated artifact independently.

    A broken primary promotes a verified standby. A broken standby leaves the
    verified primary usable and is exposed as a load diagnostic.
    """
    root = Path(bundle_dir)
    manifest_path = root / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise DemandBundleError(
            f"No serving bundle at {root}; build it with scripts/build_demand_bundle.py."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") not in {1, 2}:
        raise DemandBundleError("Unsupported demand bundle format version.")
    primary_validation = manifest.get("validation", {})
    if not isinstance(primary_validation, dict):
        raise DemandBundleError("Demand bundle is missing its primary validation report.")

    primary_error: str | None = (
        None if primary_validation.get("calibrated", False)
        else "Primary demand model did not pass its rolling-origin calibration gate."
    )
    model: Any | None = None
    if primary_error is None:
        try:
            model = _load_model_artifact(
                root,
                _MODEL_FILENAME,
                manifest.get("artifact_sha256", ""),
                manifest.get("feature_names", []),
            )
        except DemandBundleError as exc:
            primary_error = str(exc)

    backup = manifest.get("backup_model")
    backup_model: Any | None = None
    backup_manifest: dict[str, Any] | None = None
    backup_error: str | None = None
    if backup is not None:
        try:
            if backup.get("filename") != _BACKUP_MODEL_FILENAME:
                raise DemandBundleError("Backup model manifest contains an unsupported artifact path.")
            backup_validation = backup.get("validation", {})
            if not backup_validation.get("calibrated", False):
                raise DemandBundleError("Backup demand model did not pass its calibration gate.")
            candidate = manifest.get("validation", {}).get("comparison", {}).get("candidates", {}).get(
                backup.get("model_name"), {}
            )
            if not candidate.get("calibrated", False):
                raise DemandBundleError("Backup model is not marked calibrated in the comparison report.")
            backup_model = _load_model_artifact(
                root,
                _BACKUP_MODEL_FILENAME,
                backup.get("artifact_sha256", ""),
                backup.get("feature_names", []),
            )
            for shared_field in ("dataset_snapshot_id", "decision_as_of", "supported_skus"):
                if backup.get(shared_field) != manifest.get(shared_field):
                    raise DemandBundleError(
                        f"Backup model {shared_field} differs from the primary model."
                    )
            if tuple(backup.get("feature_names", ())) != tuple(manifest.get("feature_names", ())):
                raise DemandBundleError("Backup model feature schema differs from the primary model.")
            backup_manifest = {
                **manifest,
                "model_name": backup["model_name"],
                "model_version": backup["model_version"],
                "artifact_sha256": backup["artifact_sha256"],
                "feature_names": backup["feature_names"],
                "validation": backup_validation,
            }
        except (DemandBundleError, KeyError, TypeError, AttributeError) as exc:
            backup_model = None
            backup_manifest = None
            backup_error = str(exc)

    if model is not None:
        return LoadedDemandBundle(
            model=model,
            active_manifest=manifest,
            backup_model=backup_model,
            backup_manifest=backup_manifest,
            backup_error=backup_error,
        )
    if backup_model is not None and backup_manifest is not None:
        return LoadedDemandBundle(
            model=backup_model,
            active_manifest=backup_manifest,
            backup_model=None,
            backup_manifest=None,
            primary_error=primary_error,
            backup_error="Standby promoted to active model because the primary artifact failed validation.",
        )
    detail = primary_error or "Primary model artifact is unavailable."
    if backup_error:
        detail += f" Standby model is also unavailable: {backup_error}"
    raise DemandBundleError(detail)
