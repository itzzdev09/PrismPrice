"""Request-scoped serving adapter for the validated historical demand bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from prismprice.estimation.bundle import DemandBundleError, load_demand_bundle
from prismprice.features.builder import FeatureBuilder, InsufficientHistoryError
from prismprice.governance.schemas import DecisionEvidence, PriceRequest
from prismprice.serving.models import DecisionModelContext, ModelStoreUnavailable

__all__ = ["HistoricalDemandModelStore"]


@dataclass
class HistoricalDemandModelStore:
    """Serve one pinned model against a matching, point-in-time feature panel."""

    model: Any
    manifest: dict[str, Any]
    panel: pd.DataFrame

    @classmethod
    def from_bundle(cls, bundle_dir: str, panel: pd.DataFrame) -> HistoricalDemandModelStore:
        model, manifest = load_demand_bundle(bundle_dir)
        return cls(model=model, manifest=manifest, panel=panel)

    def __post_init__(self) -> None:
        supported = set(self.manifest["supported_skus"])
        data = self.panel[["sku", "date", "units", "list_price"]].copy()
        data["date"] = pd.to_datetime(data["date"], utc=True)
        data = data.dropna(subset=["sku", "date", "units", "list_price"])
        data = data[(data["units"] > 0) & (data["list_price"] > 0)]
        if data.empty:
            raise DemandBundleError("The local sales panel is empty.")

        # The bundle snapshot covers the complete source panel, including SKUs
        # that did not meet the model's training-history threshold.
        snapshot = self._panel_snapshot_id(data)
        if snapshot != self.manifest["dataset_snapshot_id"]:
            raise DemandBundleError("The active data panel differs from the model's training snapshot.")
        last_observation = pd.Timestamp(data["date"].max()).normalize()
        expected_cutoff = last_observation + pd.Timedelta(days=1)
        self.decision_as_of = expected_cutoff.to_pydatetime()
        if expected_cutoff.isoformat() != self.manifest["decision_as_of"]:
            raise DemandBundleError(
                "The local panel's last date differs from the model bundle's point-in-time cutoff."
            )

        data = data[data["sku"].astype(str).isin(supported)].copy()
        if data.empty:
            raise DemandBundleError("The model bundle's supported SKUs are absent from the data store.")
        self.builder = FeatureBuilder(
            data.rename(columns={"list_price": "price"})[["sku", "date", "units", "price"]]
        )
        self._last_observation_by_sku = (
            data.groupby(data["sku"].astype(str))["date"].max().to_dict()
        )
        self._supported_skus = supported

    @staticmethod
    def _panel_snapshot_id(panel: pd.DataFrame) -> str:
        stable = panel[["sku", "date", "units", "list_price"]].sort_values(
            ["sku", "date"]
        ).reset_index(drop=True)
        hashed_rows = pd.util.hash_pandas_object(stable, index=False, categorize=True)
        return f"sha256:{hashlib.sha256(hashed_rows.values.tobytes()).hexdigest()}"

    def demand_at(self, sku: str, price: float) -> tuple[float, float, float]:
        raise ModelStoreUnavailable("Historical model predictions require a point-in-time request context.")

    def delta_clv_at(self, sku: str, price: float) -> float:
        return 0.0

    def model_versions(self) -> dict[str, str]:
        return {"demand": self.manifest["model_version"]}

    def _features_for(
        self, request: PriceRequest
    ) -> tuple[pd.Series, dict[str, float | None], datetime, int]:
        if request.sku not in self._supported_skus:
            raise ModelStoreUnavailable(
                f"SKU {request.sku!r} is outside the validated demand model's supported catalogue."
            )
        if request.as_of != self.decision_as_of:
            raise ModelStoreUnavailable(
                "This historical model is valid only at its recorded decision date "
                f"({self.decision_as_of.isoformat()})."
            )
        try:
            vector = self.builder.build(request.sku, request.as_of)
        except InsufficientHistoryError as exc:
            raise ModelStoreUnavailable(str(exc)) from exc
        if vector.observations_used < 91:
            raise ModelStoreUnavailable("The SKU has less than 91 active sales days of history.")

        features = vector.to_series()
        features["current_price"] = request.current_price
        reference_price = float(features["reference_price_60d"])
        features["discount_depth"] = (
            1.0 - request.current_price / reference_price if reference_price > 0 else 0.0
        )
        feature_values: dict[str, float | None] = {}
        for name, value in features.items():
            feature_values[str(name)] = None if pd.isna(value) else float(value)
        last_observation = self._last_observation_by_sku[request.sku]
        return (
            features,
            feature_values,
            pd.Timestamp(last_observation).to_pydatetime(),
            vector.observations_used,
        )

    def context_for(self, request: PriceRequest) -> DecisionModelContext:
        features, feature_values, last_observation, observations_used = self._features_for(request)
        validation = self.manifest["validation"]
        evidence_without_id = {
            "sku": request.sku,
            "as_of": request.as_of.isoformat(),
            "features": feature_values,
            "dataset_snapshot_id": self.manifest["dataset_snapshot_id"],
            "model_version": self.manifest["model_version"],
        }
        feature_snapshot_id = "sha256:" + hashlib.sha256(
            json.dumps(evidence_without_id, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        evidence = DecisionEvidence(
            model_name=self.manifest["model_name"],
            model_version=self.manifest["model_version"],
            artifact_sha256=self.manifest["artifact_sha256"],
            dataset_name=self.manifest["dataset_name"],
            dataset_snapshot_id=self.manifest["dataset_snapshot_id"],
            feature_snapshot_id=feature_snapshot_id,
            feature_as_of=request.as_of,
            training_through=pd.Timestamp(self.manifest["training_through"]).to_pydatetime(),
            last_observation_at=last_observation,
            observations_used=observations_used,
            demand_unit=self.manifest["demand_unit"],
            validation_method=validation["method"],
            validation_rows=int(validation["n"]),
            validation_wape=float(validation["wape"]),
            validation_pinball_loss=float(validation["pinball_loss"]),
            validation_coverage_80=float(validation["coverage_80"]),
            validation_p10_coverage=float(validation["coverage_p10"]),
            validation_calibrated=bool(validation["calibrated"]),
            feature_values=feature_values,
            model_comparison=validation["comparison"],
        )

        def predict(price: float) -> tuple[float, float, float]:
            forecast = self.model.forecast_at_prices(features, [price])[0]
            return forecast.p10, forecast.p50, forecast.p90

        def predict_ladder(prices: list[float]) -> list[tuple[float, float, float]]:
            forecasts = self.model.forecast_at_prices(features, prices, enforce_monotonicity=True)
            return [(forecast.p10, forecast.p50, forecast.p90) for forecast in forecasts]

        return DecisionModelContext(
            demand_at=predict,
            demand_at_prices=predict_ladder,
            evidence=evidence,
        )

    def workbench_context(self) -> dict[str, Any]:
        validation = self.manifest["validation"]
        return {
            "model_store_available": True,
            "model_name": self.manifest["model_name"],
            "model_version": self.manifest["model_version"],
            "dataset_name": self.manifest["dataset_name"],
            "dataset_snapshot_id": self.manifest["dataset_snapshot_id"],
            "data_through": self.manifest["training_through"],
            "as_of": self.manifest["decision_as_of"],
            "demand_unit": self.manifest["demand_unit"],
            "supported_skus": sorted(self._supported_skus),
            "validation": {
                "method": validation["method"],
                "rows": validation["n"],
                "wape": validation["wape"],
                "pinball_loss": validation["pinball_loss"],
                "coverage_80": validation["coverage_80"],
                "coverage_p10": validation["coverage_p10"],
                "calibrated": validation["calibrated"],
                "folds": validation["folds"],
            },
            "challengers": validation["comparison"],
            "assumptions": self.manifest["assumptions"],
            "capabilities": {
                "demand_forecast": "connected",
                "causal_elasticity": "not_connected",
                "retention_clv": "not_connected",
                "portfolio_substitution": "not_connected",
                "live_costs_inventory_competitors": "not_in_source_data",
                "durable_decisions": "connected",
                "price_publishing": "not_implemented",
            },
        }
