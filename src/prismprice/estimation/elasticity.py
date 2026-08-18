"""
Causal price elasticity via Double Machine Learning (L2).

The demand model (`estimation/demand.py`) answers *how many units at this price*.
It cannot answer *what happens if we change the price*, and the distinction is
the reason this module exists. A booster fitted on logged transactions learns the
historical pricing **policy**: if the retailer discounted into strong weeks, the
booster learns that low prices coincide with high demand and will happily report
an elasticity that is mostly the buyer's calendar. Handed to an optimiser, that
number recommends discounts that do not pay for themselves.

Structural model
----------------

Demand is generated as a partially linear model, which is exactly the form DML
is built for::

    log q_it = theta_i * log p_it + g_i(X_t) + eps_it        (outcome)
    log p_it = m_i(X_t) + eta_it                             (treatment)

``theta_i`` is the own-price elasticity — the estimand. ``X`` is the observed
confounder set (seasonality, marketing spend, holiday proximity) that moves both
price and demand. ``g`` and ``m`` are unknown and non-linear, so they are fitted
with gradient boosting rather than assumed.

The estimator residualises both sides and regresses one residual on the other::

    Y_res = log q - g_hat(X)
    T_res = log p - m_hat(X)
    theta_hat = sum(T_res * Y_res) / sum(T_res ** 2)

This is the Robinson (1988) partialling-out estimator, made valid with ML
nuisance functions by Chernozhukov et al. (2018): the moment condition is Neyman
orthogonal, so first-order errors in ``g_hat`` and ``m_hat`` do not propagate
into ``theta_hat``, and cross-fitting removes the own-observation overfitting
bias that would otherwise attenuate it.

Three specification decisions
-----------------------------

**Cross-fitting folds are temporal, not random.** ``docs/architecture.md`` §4.3
specified DML via EconML, whose ``DML`` estimators cross-fit on a random K-fold
by default. On this panel that is wrong. Marketing spend is AR(1) and demand is
serially correlated, so a random fold puts day *t-1* in train and day *t* in
test; the nuisance model then predicts the test day partly by remembering its
neighbour, ``T_res`` comes out too small, and ``theta_hat`` inflates. It also
contradicts the leakage guarantee L1 is built around. Folds here are contiguous
time blocks with an optional purge gap either side of the test block.

**The estimator is implemented directly rather than delegated to EconML.** The
partialling-out estimator above is about eighty lines and is not the hard part;
the hard part is knowing whether the answer means anything. That judgement needs
the *first-stage residuals* — if ``X`` explains nearly all of the price variation
then ``T_res`` is noise, ``theta_hat`` is dividing by almost zero, and the honest
output is "not identified" rather than a number with a wide interval. EconML
does not expose those residuals stably across versions, and the confidence tag
drives degradation rung 3 (`FALLBACK_POOLED_ELASTICITY`), so it is a governance
requirement rather than a diagnostic nicety. ``tests/test_elasticity.py``
cross-checks this implementation against ``econml.dml.LinearDML`` on identical
folds, so the arithmetic is verified against the reference rather than trusted.

**Confidence is earned from residual price variation, not asserted.** The spec
tags an estimate `high` when "genuine experimental or uncorrelated price
variation exists". Operationally that is the spread of ``T_res``: price movement
that the confounders do *not* explain. It is reported on every estimate, and an
estimate below the floor is tagged `low` whatever its interval looks like.

Censoring
---------

Pass un-censored demand. On stockout days the recorded units are
``min(demand, stock)``, so raw units make demand look like it collapses exactly
when a product sells well — which attenuates ``theta`` toward zero. Fit
:class:`prismprice.features.uncensoring.TobitUncensoring` first and pass its
output column. ``tests/test_elasticity.py`` measures the attenuation rather than
asserting the warning is worth heeding.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from prismprice import config
from prismprice.compute import lightgbm_device_params

__all__ = [
    "DoubleMLElasticity",
    "ElasticityEstimate",
    "ElasticityScore",
    "add_category_price_control",
    "naive_ols_elasticity",
    "score_against_truth",
    "temporal_folds",
]

#: Normal quantile for a 95% two-sided interval. Hardcoding this rather than
#: importing scipy keeps the module's dependency surface to numpy and LightGBM;
#: `confidence_level` other than 0.95 resolves through scipy when available.
_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class ElasticityEstimate:
    """One SKU's causal own-price elasticity, with the evidence for it.

    ``confidence`` is the field the decision layer reads. ``low`` means the
    number should not be used on its own — L3 degrades to the pooled category
    elasticity (rung 3, ``FALLBACK_POOLED_ELASTICITY``) rather than optimising
    against it.
    """

    sku: str
    point: float
    ci_low: float
    ci_high: float
    std_error: float
    method: str
    confidence: str
    reason: str
    n_observations: int
    n_dropped: int
    residual_price_sd: float
    """Spread of log price *after* partialling out confounders. This is what
    identifies the elasticity; near zero means the data cannot answer."""
    price_variation_explained: float
    """R^2 of the first-stage price model. High means little exogenous variation
    is left, and the estimate is being read off noise."""

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def is_usable(self) -> bool:
        return self.confidence == "high"

    def contains(self, value: float) -> bool:
        """Whether *value* lies inside the confidence interval."""
        return self.ci_low <= value <= self.ci_high

    def as_dict(self) -> dict[str, Any]:
        """Serialisation matching the `estimates.causal_elasticity` block of the
        decision record (architecture.md §5)."""
        return {
            "sku": self.sku,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std_error": self.std_error,
            "method": self.method,
            "confidence": self.confidence,
            "reason": self.reason,
            "n_observations": self.n_observations,
            "n_dropped": self.n_dropped,
            "residual_price_sd": self.residual_price_sd,
            "price_variation_explained": self.price_variation_explained,
        }


@dataclass(frozen=True)
class ElasticityScore:
    """Recovery of known ground truth. The phase-3 gate reads ``coverage``."""

    n_skus: int
    coverage: float
    """Fraction of SKUs whose true elasticity falls inside the estimated CI.
    The gate is >= 0.90; a well-calibrated 95% interval should reach ~0.95."""
    mean_bias: float
    rmse: float
    mean_ci_width: float
    n_high_confidence: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_skus": self.n_skus,
            "coverage": self.coverage,
            "mean_bias": self.mean_bias,
            "rmse": self.rmse,
            "mean_ci_width": self.mean_ci_width,
            "n_high_confidence": self.n_high_confidence,
        }


def temporal_folds(
    n_rows: int, n_folds: int, purge: int = 0, offset: int = 0
) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
    """Contiguous time-block folds for cross-fitting.

    Each fold's test block is a contiguous slice of the (time-ordered) rows, and
    its training set is everything outside that slice **plus a purge margin**.
    The margin matters because serial correlation makes the rows immediately
    adjacent to the test block nearly as informative as the block itself, which
    is precisely the leakage a random K-fold would allow everywhere.

    Args:
        n_rows: Number of time-ordered observations.
        n_folds: Number of blocks, at least 2.
        purge: Rows dropped from training either side of the test block.
        offset: Shifts the interior cut points, so repeated cross-fitting can
            draw a genuinely different partition while every block stays
            contiguous in time.

    Returns:
        ``(train_index, test_index)`` pairs, test blocks in ascending time order.

    Raises:
        ValueError: on fewer than 2 folds, or too few rows to form them.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2 for cross-fitting, got {n_folds}")
    if n_rows < n_folds * 2:
        raise ValueError(f"need at least {n_folds * 2} rows for {n_folds} folds, got {n_rows}")

    interior = [int(b) for b in np.linspace(0, n_rows, n_folds + 1)[1:-1]]
    shifted = {min(max(b + offset, 1), n_rows - 1) for b in interior}
    bounds = sorted({0, n_rows} | shifted)
    all_rows = np.arange(n_rows, dtype=np.int64)
    folds: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []

    for i in range(len(bounds) - 1):
        start, stop = int(bounds[i]), int(bounds[i + 1])
        test = all_rows[start:stop]
        blocked_lo = max(0, start - purge)
        blocked_hi = min(n_rows, stop + purge)
        mask = np.ones(n_rows, dtype=bool)
        mask[blocked_lo:blocked_hi] = False
        train = all_rows[mask]
        if train.size == 0:
            raise ValueError(
                f"purge={purge} leaves fold {i} with no training rows; "
                f"reduce purge or n_folds"
            )
        folds.append((train, test))

    return folds


@dataclass
class DoubleMLElasticity:
    """Cross-fitted partially-linear DML estimator of own-price elasticity.

    Args:
        confounder_columns: Observed variables that move both price and demand.
            Anything omitted here that does both is an unobserved confounder and
            biases the result — see README §10.
        price_column: Treatment, entered as ``log(price)``.
        outcome_column: Demand. Pass an **un-censored** series; see module
            docstring.
        n_folds: Cross-fitting blocks.
        n_repeats: Independent fold partitions to aggregate over. ``1`` gives
            textbook single-split DML, whose interval is measurably too narrow
            here; see :meth:`_estimate_one`.
        purge: Rows excluded either side of each test block.
        max_ci_width: ``tau_max``. Wider than this and the estimate is tagged
            ``low`` regardless of the point value.
        min_residual_price_sd: Identification floor on the spread of residual
            log price. Below it the estimate is tagged ``low``.
        confidence_level: Two-sided interval level.
        hac_bandwidth: Newey-West lag truncation. ``None`` uses the automatic
            ``4(n/100)^(2/9)`` rule; ``0`` recovers the iid variance, which is
            anti-conservative on serially correlated panels.
    """

    confounder_columns: tuple[str, ...] = (
        "season_yearly",
        "is_weekend",
        "marketing_spend",
        "holiday",
    )
    price_column: str = "price"
    outcome_column: str = "units"
    sku_column: str = "sku"
    date_column: str = "date"
    n_folds: int = 5
    n_repeats: int = 5
    purge: int = 3
    max_ci_width: float = 1.0
    min_residual_price_sd: float = 0.02
    confidence_level: float = 0.95
    hac_bandwidth: int | None = None
    n_estimators: int = 200
    learning_rate: float = 0.03
    num_leaves: int = 4
    min_child_samples: int = 30
    seed: int = config.DEFAULT_SEED

    estimates: dict[str, ElasticityEstimate] = field(default_factory=dict, init=False)
    device_params: dict[str, Any] = field(default_factory=dict, init=False)

    METHOD = "dml-partialling-out"

    # -- public API --------------------------------------------------------

    def fit(self, daily: pd.DataFrame) -> DoubleMLElasticity:
        """Estimate elasticity for every SKU in *daily*.

        Args:
            daily: Panel with one row per ``(sku, date)``.

        Returns:
            Self, with :attr:`estimates` populated.

        Raises:
            ValueError: on missing required columns.
        """
        self._validate(daily)
        _warn_if_censored(daily, self.outcome_column)
        # Resolved once rather than per SKU per fold, so a CPU fallback is
        # reported once instead of several hundred times.
        self.device_params = lightgbm_device_params("estimation.elasticity")

        self.estimates = {}
        for sku, frame in daily.groupby(self.sku_column, sort=True):
            ordered = frame.sort_values(self.date_column)
            self.estimates[str(sku)] = self._estimate_one(str(sku), ordered)
        return self

    def pooled(self, daily: pd.DataFrame) -> ElasticityEstimate:
        """Single category-wide elasticity, for degradation rung 3.

        **Pools the residuals, not the rows.** The obvious implementation — throw
        every SKU into one fit with SKU identity as a feature — is wrong, and
        wrong in the direction that looks plausible. SKUs differ in baseline
        volume and base price, so stacking them creates cross-sectional spread in
        ``(log p, log q)`` that has nothing to do with anyone responding to a
        price change. A boosted tree cannot absorb that from an integer SKU code
        the way a fixed effect would, and the contamination drags the estimate
        toward zero: measured on a synthetic panel whose true mean elasticity was
        -1.99, the stacked version returned -0.82.

        Each SKU is therefore residualised against its own nuisance models, which
        absorbs its level exactly, and the pooled elasticity is the aggregate
        moment over the stacked residuals::

            theta = sum_i sum_t (T_res_it * Y_res_it) / sum_i sum_t T_res_it^2

        This is the number L3 falls back to when a SKU's own estimate is tagged
        ``low``, so being quietly attenuated would mean degrading to a *more*
        confident-looking wrong answer.
        """
        self._validate(daily)
        _warn_if_censored(daily, self.outcome_column)
        if not self.device_params:
            self.device_params = lightgbm_device_params("estimation.elasticity")

        per_repeat_theta: list[float] = []
        per_repeat_variance: list[float] = []
        residual_sds: list[float] = []
        total_rows = 0
        total_dropped = 0

        prepared = []
        for _, frame in daily.groupby(self.sku_column, sort=True):
            ordered = frame.sort_values(self.date_column)
            fields = self._prepare(ordered)
            if fields is None:
                continue
            prepared.append(fields)
            total_rows += fields[3]
            total_dropped += fields[4]

        if not prepared:
            return self._unidentified(
                "__pooled__", self.METHOD + "-pooled", total_rows, total_dropped,
                "no SKU had enough usable rows to pool",
            )

        for repeat in range(max(1, self.n_repeats)):
            stacked_t: list[NDArray[np.float64]] = []
            stacked_y: list[NDArray[np.float64]] = []
            for confounders, treatment, response, n_rows, _ in prepared:
                offset = int(repeat * n_rows / (self.n_folds * max(1, self.n_repeats)))
                t_res, y_res = self._cross_fitted_residuals(
                    confounders, treatment, response, offset
                )
                stacked_t.append(t_res)
                stacked_y.append(y_res)

            t_all = np.concatenate(stacked_t)
            y_all = np.concatenate(stacked_y)
            denominator = float(np.sum(t_all**2))
            if denominator <= 0:
                continue
            theta = float(np.sum(t_all * y_all) / denominator)
            psi = t_all * (y_all - theta * t_all)
            jacobian = float(np.mean(t_all**2))
            per_repeat_theta.append(theta)
            per_repeat_variance.append(
                float(np.mean(psi**2)) / (jacobian**2) / t_all.shape[0]
            )
            residual_sds.append(float(np.std(t_all)))

        if not per_repeat_theta:
            return self._unidentified(
                "__pooled__", self.METHOD + "-pooled", total_rows, total_dropped,
                "no residual price variation after partialling out confounders",
            )

        theta_array = np.array(per_repeat_theta, dtype=float)
        variance_array = np.array(per_repeat_variance, dtype=float)
        theta = float(np.median(theta_array))
        variance = float(np.median(variance_array + (theta_array - theta) ** 2))
        std_error = float(np.sqrt(variance))
        z = self._z_value()
        residual_price_sd = float(np.median(residual_sds))
        confidence, reason = self._tag(theta, 2 * z * std_error, residual_price_sd)

        return ElasticityEstimate(
            sku="__pooled__",
            point=theta,
            ci_low=theta - z * std_error,
            ci_high=theta + z * std_error,
            std_error=std_error,
            method=self.METHOD + "-pooled",
            confidence=confidence,
            reason=reason,
            n_observations=total_rows,
            n_dropped=total_dropped,
            residual_price_sd=residual_price_sd,
            price_variation_explained=float("nan"),
        )

    # -- internals ---------------------------------------------------------

    def _validate(self, daily: pd.DataFrame) -> None:
        required = {
            self.sku_column,
            self.date_column,
            self.price_column,
            self.outcome_column,
            *self.confounder_columns,
        }
        missing = sorted(required - set(daily.columns))
        if missing:
            raise ValueError(f"Panel is missing required columns: {missing}")

    def _z_value(self) -> float:
        if abs(self.confidence_level - 0.95) < 1e-12:
            return _Z_95
        from scipy import stats  # optional; only needed off the default level

        return float(stats.norm.ppf(0.5 + self.confidence_level / 2.0))

    def _prepare(
        self,
        frame: pd.DataFrame,
        extra_features: dict[str, NDArray[np.float64]] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int, int] | None:
        """Build the design arrays for one SKU, or ``None`` if too thin to fit.

        Returns ``(confounders, log_price, log_demand, n_usable, n_dropped)``.
        Rows whose price or demand is non-positive are dropped and counted: the
        log of a non-positive quantity is undefined, and substituting ``log1p``
        would quietly replace the structural model with a different one.
        """
        price = frame[self.price_column].to_numpy(dtype=float)
        outcome = frame[self.outcome_column].to_numpy(dtype=float)

        usable = (price > 0) & (outcome > 0) & np.isfinite(price) & np.isfinite(outcome)
        n = int(usable.sum())
        n_dropped = int((~usable).sum())

        if n < self.n_folds * max(2, self.min_child_samples // 2):
            return None

        columns = [frame[c].to_numpy(dtype=float) for c in self.confounder_columns]
        if extra_features:
            columns += list(extra_features.values())
        confounders = np.column_stack(columns)[usable]

        # A control with missing values (a single-SKU day has no rest-of-category
        # price) would silently poison the nuisance fit, so it is mean-filled and
        # the fill is visible here rather than buried in the learner.
        if not np.all(np.isfinite(confounders)):
            column_means = np.nanmean(np.where(np.isfinite(confounders), confounders, np.nan), axis=0)
            column_means = np.where(np.isfinite(column_means), column_means, 0.0)
            confounders = np.where(np.isfinite(confounders), confounders, column_means)

        return confounders, np.log(price[usable]), np.log(outcome[usable]), n, n_dropped

    def _estimate_one(
        self,
        sku: str,
        frame: pd.DataFrame,
        extra_features: dict[str, NDArray[np.float64]] | None = None,
        method_suffix: str = "",
    ) -> ElasticityEstimate:
        method = self.METHOD + method_suffix
        fields = self._prepare(frame, extra_features)
        if fields is None:
            usable_rows = int(
                (
                    (frame[self.price_column].to_numpy(dtype=float) > 0)
                    & (frame[self.outcome_column].to_numpy(dtype=float) > 0)
                ).sum()
            )
            return self._unidentified(
                sku, method, usable_rows, len(frame) - usable_rows,
                f"only {usable_rows} usable rows for {self.n_folds}-fold cross-fitting",
            )
        confounders, treatment, response, n, n_dropped = fields

        # Repeated cross-fitting. A single fold partition gives an interval that
        # is measurably too narrow: on synthetic panels the estimates scattered
        # 1.38x wider than their own standard errors claimed, so a nominal 95%
        # interval covered 83% of true elasticities. The point estimate was
        # near-unbiased throughout — the defect was entirely in the variance.
        #
        # The missing term is the uncertainty from having estimated the nuisance
        # functions at all. The asymptotic formula is derived as if g and m were
        # known, and an oracle fit that *does* know them lands at exactly the
        # standard error reported here (0.080 against a nominal 0.086), which is
        # what identifies nuisance error rather than serial correlation as the
        # cause. Chernozhukov et al. (2018) §3.4 handle it by repeating the fit
        # over independent partitions and aggregating with the between-split
        # spread folded into the variance:
        #
        #     theta = median_s(theta_s)
        #     var   = median_s(var_s + (theta_s - theta)^2)
        #
        # That recovers 0.944 coverage across five seeds, with no seed below 0.92.
        thetas: list[float] = []
        variances: list[float] = []
        residual_sds: list[float] = []

        for repeat in range(max(1, self.n_repeats)):
            offset = int(repeat * n / (self.n_folds * max(1, self.n_repeats)))
            t_res, y_res = self._cross_fitted_residuals(
                confounders, treatment, response, offset
            )
            fold_denominator = float(np.sum(t_res**2))
            if fold_denominator <= 0:
                continue
            fold_theta = float(np.sum(t_res * y_res) / fold_denominator)
            psi = t_res * (y_res - fold_theta * t_res)
            jacobian = float(np.mean(t_res**2))
            thetas.append(fold_theta)
            variances.append(self._long_run_variance(psi) / (jacobian**2) / n)
            residual_sds.append(float(np.std(t_res)))

        treatment_var = float(np.var(treatment))
        if not thetas:
            return self._unidentified(
                sku, method, n, n_dropped,
                "no residual price variation after partialling out confounders",
            )

        residual_price_sd = float(np.median(residual_sds))
        explained = (
            float(np.clip(1.0 - (residual_price_sd**2) / treatment_var, 0.0, 1.0))
            if treatment_var > 0
            else 1.0
        )

        if residual_price_sd < self.min_residual_price_sd:
            return self._unidentified(
                sku, method, n, n_dropped,
                f"residual log-price SD {residual_price_sd:.4f} is below the identification "
                f"floor {self.min_residual_price_sd:.4f}",
                residual_price_sd=residual_price_sd,
                explained=explained,
            )

        theta_array = np.array(thetas, dtype=float)
        variance_array = np.array(variances, dtype=float)
        theta = float(np.median(theta_array))
        variance = float(np.median(variance_array + (theta_array - theta) ** 2))
        std_error = float(np.sqrt(variance))

        z = self._z_value()
        ci_low, ci_high = theta - z * std_error, theta + z * std_error

        confidence, reason = self._tag(theta, ci_high - ci_low, residual_price_sd)

        return ElasticityEstimate(
            sku=sku,
            point=theta,
            ci_low=ci_low,
            ci_high=ci_high,
            std_error=std_error,
            method=method,
            confidence=confidence,
            reason=reason,
            n_observations=n,
            n_dropped=n_dropped,
            residual_price_sd=residual_price_sd,
            price_variation_explained=explained,
        )

    def _long_run_variance(self, psi: NDArray[np.float64]) -> float:
        """Newey-West long-run variance of the orthogonal score.

        The textbook DML variance is ``E[psi^2]``, which assumes the score is
        serially uncorrelated. On a daily panel that assumption is false —
        demand is autocorrelated and marketing spend is AR(1) — so the Bartlett
        correction is the right specification and is applied here.

        **It was not, however, the fix for the narrow intervals.** The intervals
        were 34% too tight, serial correlation was the obvious suspect, and this
        correction moved the standard error by roughly 4%. Measuring that ruled
        the suspect out and pointed at nuisance-estimation error instead, which
        repeated cross-fitting in :meth:`_estimate_one` addresses. The kernel is
        kept because it is correct, not because it earned its keep; the comment
        records which of the two actually mattered so the next person does not
        re-run the same dead end.

        Bartlett kernel, bandwidth ``4(n/100)^(2/9)`` (Newey & West 1994).

        Args:
            psi: Score contributions **in time order**. Order is load-bearing;
                the caller sorts by date and the usable-row mask preserves it.

        Returns:
            The long-run variance estimate, never below the iid value.
        """
        n = psi.shape[0]
        centred = psi - psi.mean()
        gamma_0 = float(np.mean(centred**2))

        if self.hac_bandwidth is not None:
            bandwidth = self.hac_bandwidth
        else:
            bandwidth = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        bandwidth = max(0, min(bandwidth, n - 1))

        total = gamma_0
        for lag in range(1, bandwidth + 1):
            gamma_l = float(np.mean(centred[lag:] * centred[:-lag]))
            weight = 1.0 - lag / (bandwidth + 1.0)
            total += 2.0 * weight * gamma_l

        # Negative autocorrelation can in principle drive the sum below the iid
        # variance. Shrinking the interval on that basis would be claiming the
        # serial structure buys precision, which the Bartlett kernel does not
        # establish, so the iid value is a floor.
        return max(total, gamma_0)

    def _cross_fitted_residuals(
        self,
        confounders: NDArray[np.float64],
        treatment: NDArray[np.float64],
        response: NDArray[np.float64],
        offset: int = 0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Out-of-fold residuals for both stages.

        Every prediction is made by a model that never saw the row (nor, thanks
        to the purge, its immediate temporal neighbours), which is what removes
        the overfitting bias that would otherwise shrink ``T_res`` and inflate
        the elasticity.
        """
        n = treatment.shape[0]
        t_res = np.zeros(n, dtype=float)
        y_res = np.zeros(n, dtype=float)

        for train_idx, test_idx in temporal_folds(n, self.n_folds, self.purge, offset):
            price_model = self._learner()
            price_model.fit(confounders[train_idx], treatment[train_idx])
            t_res[test_idx] = treatment[test_idx] - price_model.predict(confounders[test_idx])

            demand_model = self._learner()
            demand_model.fit(confounders[train_idx], response[train_idx])
            y_res[test_idx] = response[test_idx] - demand_model.predict(confounders[test_idx])

        return t_res, y_res

    def _learner(self) -> Any:
        """Nuisance regressor.

        Boosted trees rather than a linear model on purpose: the confounding in
        this problem runs through a *threshold* (a promotion fires when demand
        pressure crosses a level), so ``E[log p | X]`` is non-linear and a linear
        first stage would leave exactly the confounding DML is meant to remove.
        """
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            random_state=self.seed,
            verbose=-1,
            **self.device_params,
        )

    def _tag(self, theta: float, ci_width: float, residual_price_sd: float) -> tuple[str, str]:
        """Assign the confidence tag L3 reads, and say why."""
        if residual_price_sd < self.min_residual_price_sd:
            return "low", (
                f"weak identification: residual log-price SD {residual_price_sd:.4f} is below "
                f"the floor {self.min_residual_price_sd:.4f}; confounders explain almost all "
                f"price movement, so the elasticity is not identified from this data"
            )
        if ci_width > self.max_ci_width:
            return "low", (
                f"interval width {ci_width:.3f} exceeds tau_max {self.max_ci_width:.3f}"
            )
        if theta >= 0:
            return "low", (
                f"estimated elasticity {theta:.3f} is non-negative, which contradicts "
                f"downward-sloping demand; treat as a model failure, not a finding"
            )
        return "high", "identified from residual price variation; interval within tau_max"

    def _unidentified(
        self,
        sku: str,
        method: str,
        n: int,
        n_dropped: int,
        reason: str,
        residual_price_sd: float = 0.0,
        explained: float = 1.0,
    ) -> ElasticityEstimate:
        """An estimate that declines to be one.

        Returns NaN rather than a number, because a caller that sees 0.0 cannot
        tell "no effect" from "no answer", and only the second should degrade.
        """
        return ElasticityEstimate(
            sku=sku,
            point=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            std_error=float("nan"),
            method=method,
            confidence="low",
            reason=reason,
            n_observations=n,
            n_dropped=n_dropped,
            residual_price_sd=residual_price_sd,
            price_variation_explained=explained,
        )

    def as_frame(self) -> pd.DataFrame:
        """Estimates as a frame, joinable to ground truth for scoring."""
        if not self.estimates:
            raise ValueError("fit() has not been called")
        return pd.DataFrame([e.as_dict() for e in self.estimates.values()])


def add_category_price_control(
    daily: pd.DataFrame,
    price_column: str = "price",
    date_column: str = "date",
    output_column: str = "category_log_price",
) -> pd.DataFrame:
    """Add the leave-one-out mean log price of the rest of the category.

    Substitutes inside a category share buyers, so a rival SKU's discount moves
    this SKU's demand. Left out of the confounder set that variation lands in the
    error term, and because every SKU's promotion calendar responds to the same
    demand pressure, it does not average away — it widens the scatter of the
    estimates without widening their intervals.

    Adding it is not a synthetic-data trick: a retailer observes its own category
    prices, so this is a control that exists in production. Measured across five
    seeds it moved coverage from 0.832 to 0.880 and cut the excess scatter from
    1.38x its nominal standard error to 1.21x, before repeated cross-fitting
    closed the rest.

    Leave-one-out matters. Including this SKU's own price in the category mean
    would put the treatment on both sides of the regression and bias the
    elasticity toward zero.

    Args:
        daily: Panel with one row per ``(sku, date)``.
        price_column: Price to average.
        date_column: Column defining "the same day".
        output_column: Name for the added control.

    Returns:
        A copy of *daily* with ``output_column`` added. Days holding a single
        SKU get ``NaN``, since there is no rest-of-category to average.
    """
    frame = daily.copy()
    log_price = np.log(frame[price_column].astype(float))
    grouped = log_price.groupby(frame[date_column])
    total = grouped.transform("sum")
    count = grouped.transform("count")
    frame[output_column] = (total - log_price) / (count - 1).replace(0, np.nan)
    return frame


def naive_ols_elasticity(
    daily: pd.DataFrame,
    price_column: str = "price",
    outcome_column: str = "units",
    sku_column: str = "sku",
) -> dict[str, float]:
    """Per-SKU elasticity from an uncontrolled log-log regression.

    The straw man, and the reason this layer exists. It omits every confounder,
    so on a panel where the retailer discounted into strong weeks it attributes
    the calendar's demand to the discount and overstates how elastic buyers are.
    Kept in the package rather than in the tests because L3's decision record
    reports it alongside the causal estimate: the gap between the two is the
    clearest single statement of what confounding was worth.
    """
    out: dict[str, float] = {}
    for sku, frame in daily.groupby(sku_column, sort=True):
        price = frame[price_column].to_numpy(dtype=float)
        units = frame[outcome_column].to_numpy(dtype=float)
        usable = (price > 0) & (units > 0)
        if usable.sum() < 3:
            out[str(sku)] = float("nan")
            continue
        x = np.log(price[usable])
        y = np.log(units[usable])
        x_centred = x - x.mean()
        denominator = float(np.sum(x_centred**2))
        out[str(sku)] = (
            float(np.sum(x_centred * (y - y.mean())) / denominator)
            if denominator > 0
            else float("nan")
        )
    return out


def score_against_truth(
    estimates: dict[str, ElasticityEstimate],
    truth: pd.DataFrame,
    truth_column: str = "true_beta",
    sku_column: str = "sku",
) -> ElasticityScore:
    """Score recovered elasticities against known generator parameters.

    Scores **every** SKU, including those the estimator declined to answer for.
    Excluding them would let a model reach perfect coverage by refusing to
    answer whenever it was unsure, which is the failure mode a coverage gate is
    supposed to catch.
    """
    truth_map = dict(
        zip(truth[sku_column].astype(str), truth[truth_column].astype(float), strict=False)
    )

    covered = 0
    errors: list[float] = []
    widths: list[float] = []
    high = 0

    for sku, estimate in estimates.items():
        if sku not in truth_map:
            continue
        true_beta = truth_map[sku]
        if np.isfinite(estimate.point):
            if estimate.contains(true_beta):
                covered += 1
            errors.append(estimate.point - true_beta)
            widths.append(estimate.ci_width)
        if estimate.is_usable:
            high += 1

    n = len([s for s in estimates if s in truth_map])
    if n == 0:
        raise ValueError("no SKUs in common between estimates and truth")

    error_array = np.array(errors, dtype=float) if errors else np.array([np.nan])
    return ElasticityScore(
        n_skus=n,
        coverage=covered / n,
        mean_bias=float(np.mean(error_array)),
        rmse=float(np.sqrt(np.mean(error_array**2))),
        mean_ci_width=float(np.mean(widths)) if widths else float("nan"),
        n_high_confidence=high,
    )


def _warn_if_censored(daily: pd.DataFrame, outcome_column: str) -> None:
    """Warn when elasticity is being fitted on a visibly censored series."""
    if "stockout" in daily.columns and outcome_column == "units":
        censored = float(daily["stockout"].mean())
        if censored > 0.0:
            warnings.warn(
                f"Fitting elasticity on raw 'units' with {censored:.1%} stockout days. "
                f"Censored demand attenuates elasticity toward zero; fit "
                f"TobitUncensoring first and pass its output column.",
                RuntimeWarning,
                stacklevel=3,
            )
