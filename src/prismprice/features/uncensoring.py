"""
Demand un-censoring (L1).

During a stockout, observed sales are not demand — they are the smaller of
demand and available stock. Training a demand model on the raw series teaches it
that demand collapses exactly when the product sells well, which then feeds a
price recommendation built on a number that was never true.

**Right-censored, not censored at zero.** ``docs/data-and-modelling.md`` framed
this as a Type I Tobit censored at 0, which is the textbook case where a
stockout means zero recorded sales. Real replenishment produces *partial*
fulfilment: stock runs out mid-period having served some units, so all that is
known is ``demand >= units_served``. That is right-censoring at the observed
value, and censoring at zero would be the wrong likelihood for it.

The model is fitted in log space because the generator — and demand generally —
is multiplicative: ``log q = x'B + e`` with normal ``e``. In levels the
normality assumption the Tobit likelihood rests on would simply be false.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import optimize, stats

__all__ = [
    "TobitUncensoring",
    "UncensoringScore",
    "score_uncensoring",
]

_MIN_UNITS = 1e-3


@dataclass
class TobitUncensoring:
    """Right-censored Tobit estimator of latent demand.

    Args:
        feature_columns: Design-matrix columns. SKU identity is added as
            fixed effects automatically when ``sku_fixed_effects`` is set.
        censored_column: Boolean column marking rows where stock bound.
        units_column: Observed (possibly censored) units.
        sku_fixed_effects: Fit a per-SKU intercept. Without it a single
            intercept is shared across SKUs of wildly different baseline
            volume, and the censoring correction inherits that misfit.
    """

    feature_columns: tuple[str, ...] = (
        "log_price",
        "season_yearly",
        "is_weekend",
        "marketing_spend",
        "holiday",
    )
    censored_column: str = "stockout"
    units_column: str = "units"
    sku_fixed_effects: bool = True

    coefficients: NDArray[np.float64] | None = field(default=None, init=False)
    sigma: float | None = field(default=None, init=False)
    design_columns: tuple[str, ...] = field(default=(), init=False)
    converged: bool = field(default=False, init=False)

    # -- design -----------------------------------------------------------

    def _design(self, df: pd.DataFrame, *, fit: bool) -> NDArray[np.float64]:
        frame = pd.DataFrame(index=df.index)
        frame["intercept"] = 1.0

        for column in self.feature_columns:
            if column == "log_price" and "log_price" not in df.columns:
                frame["log_price"] = np.log(df["price"].to_numpy(dtype=float))
            else:
                frame[column] = df[column].to_numpy(dtype=float)

        if self.sku_fixed_effects and "sku" in df.columns:
            dummies = pd.get_dummies(df["sku"], prefix="sku", drop_first=True, dtype=float)
            frame = pd.concat([frame, dummies], axis=1)

        if fit:
            self.design_columns = tuple(frame.columns)
        else:
            # Align to the fitted design: unseen SKUs contribute no fixed effect
            # rather than silently shifting every other column by one position.
            frame = frame.reindex(columns=list(self.design_columns), fill_value=0.0)

        design: NDArray[np.float64] = frame.to_numpy(dtype=float)
        return design

    # -- likelihood -------------------------------------------------------

    @staticmethod
    def _negative_log_likelihood(
        params: NDArray[np.float64],
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        censored: NDArray[np.bool_],
    ) -> float:
        beta, log_sigma = params[:-1], params[-1]
        sigma = float(np.exp(log_sigma))
        residual = (y - x @ beta) / sigma

        uncensored_ll = np.where(censored, 0.0, stats.norm.logpdf(residual) - log_sigma)
        # P(latent > observed) for the rows where stock bound.
        censored_ll = np.where(censored, stats.norm.logsf(residual), 0.0)

        total = float(np.sum(uncensored_ll) + np.sum(censored_ll))
        return -total if np.isfinite(total) else 1e12

    def fit(self, df: pd.DataFrame) -> TobitUncensoring:
        """Fit by maximum likelihood. Raises if the panel cannot identify a fit."""
        censored = df[self.censored_column].to_numpy(dtype=bool)
        if censored.all():
            raise ValueError(
                "Every observation is censored; there is no uncensored variation "
                "to identify the demand equation from"
            )

        x = self._design(df, fit=True)
        y = np.log(np.maximum(df[self.units_column].to_numpy(dtype=float), _MIN_UNITS))

        # OLS on the uncensored rows is the natural start: it is the MLE when
        # nothing is censored, so the optimiser begins in the right basin.
        open_rows = ~censored
        start_beta, *_ = np.linalg.lstsq(x[open_rows], y[open_rows], rcond=None)
        residual_sd = float(np.std(y[open_rows] - x[open_rows] @ start_beta)) or 1.0
        start = np.append(start_beta, np.log(residual_sd))

        result = optimize.minimize(
            self._negative_log_likelihood,
            start,
            args=(x, y, censored),
            method="L-BFGS-B",
            options={"maxiter": 2_000},
        )

        self.coefficients = result.x[:-1]
        self.sigma = float(np.exp(result.x[-1]))
        self.converged = bool(result.success)
        return self

    # -- prediction -------------------------------------------------------

    def expected_latent_log_demand(self, df: pd.DataFrame) -> NDArray[np.float64]:
        """``E[log q* | observation]``, applying the truncation correction.

        For a censored row the estimate is ``x'B + sigma * lambda(a)`` where
        ``lambda`` is the inverse Mills ratio — the mean of the normal tail above
        the point where stock ran out. Uncensored rows keep their observed value:
        the model exists to fill gaps, not to overwrite facts.
        """
        if self.coefficients is None or self.sigma is None:
            raise RuntimeError("TobitUncensoring must be fitted before prediction")

        x = self._design(df, fit=False)
        censored = df[self.censored_column].to_numpy(dtype=bool)
        observed_log = np.log(np.maximum(df[self.units_column].to_numpy(dtype=float), _MIN_UNITS))

        linear = x @ self.coefficients
        alpha = (observed_log - linear) / self.sigma
        mills = stats.norm.pdf(alpha) / np.clip(stats.norm.sf(alpha), 1e-12, None)
        truncated_mean = linear + self.sigma * mills

        # The conditional mean of the tail can never sit below the point the tail
        # starts at; clip guards the far-tail region where the ratio is numerically
        # unstable.
        truncated_mean = np.maximum(truncated_mean, observed_log)
        latent: NDArray[np.float64] = np.where(censored, truncated_mean, observed_log)
        return latent

    def transform(
        self, df: pd.DataFrame, output_column: str = "units_uncensored_est"
    ) -> pd.DataFrame:
        """Return *df* with an added estimate of latent demand in units."""
        out = df.copy()
        out[output_column] = np.exp(self.expected_latent_log_demand(df))
        return out

    def fit_transform(
        self, df: pd.DataFrame, output_column: str = "units_uncensored_est"
    ) -> pd.DataFrame:
        return self.fit(df).transform(df, output_column=output_column)


@dataclass(frozen=True)
class UncensoringScore:
    """Recovery quality on censored rows, versus doing nothing."""

    censored_rows: int
    naive_wape: float
    """WAPE of the raw observed series against latent truth, on censored rows."""
    estimated_wape: float
    """WAPE of the un-censored estimate against latent truth, on censored rows."""

    @property
    def improvement(self) -> float:
        """Share of the naive error removed. Negative means it made things worse."""
        if self.naive_wape == 0:
            return 0.0
        return (self.naive_wape - self.estimated_wape) / self.naive_wape


def score_uncensoring(
    df: pd.DataFrame,
    truth_column: str = "units_uncensored",
    estimate_column: str = "units_uncensored_est",
    units_column: str = "units",
    censored_column: str = "stockout",
) -> UncensoringScore:
    """Score an un-censoring estimate against known latent demand.

    Only censored rows are scored. Uncensored rows are passed through unchanged
    by construction, so including them would dilute the metric toward zero and
    make a useless estimator look good.
    """
    censored = df[df[censored_column].astype(bool)]
    if censored.empty:
        return UncensoringScore(censored_rows=0, naive_wape=0.0, estimated_wape=0.0)

    truth = censored[truth_column].to_numpy(dtype=float)
    denominator = float(np.sum(np.abs(truth))) or 1.0
    naive = float(np.sum(np.abs(censored[units_column].to_numpy(dtype=float) - truth)))
    estimated = float(np.sum(np.abs(censored[estimate_column].to_numpy(dtype=float) - truth)))

    return UncensoringScore(
        censored_rows=len(censored),
        naive_wape=naive / denominator,
        estimated_wape=estimated / denominator,
    )
