"""
Retention, repurchase hazard and Delta-CLV (L2, phase 4).

The layer that makes this a *pricing* system rather than a margin calculator. A
price does two things: it settles today's transaction, and it moves the
probability that the customer comes back. Optimise only the first and the
system learns that the best price is always the highest one the guardrails
allow, because the cost of the discount is visible today and the cost of the
lost customer is not.

Model
-----

Repurchase timing is a Cox proportional-hazards model on the gap between
purchases, with the paid price entering as a **shock relative to reference**::

    h(t | x) = h_0(t) * exp(eta(x)),    eta(x) = theta * (p / p_ref - 1) + ...

Two properties of that specification are doing real work.

**The covariate is the price shock, not the price.** An absolute price cannot
explain churn — GBP 40 is cheap for one product and insulting for another. What
a customer reacts to is paying more than they expected to, which is why the
generator builds churn off ``p / p_ref - 1`` and why the estimator must read the
same quantity. Fitting on raw price would produce a model that says expensive
products have loyal customers, because expensive products *do*, for reasons that
have nothing to do with pricing.

**The baseline hazard is left unspecified.** Cox estimates ``theta`` from the
*order* in which customers return, never from the shape of ``h_0``. That matters
here because the shape is unknown and misspecifying it biases the coefficient
that the objective actually consumes. A Weibull AFT would be more efficient if
the Weibull assumption held, and silently wrong if it did not.

``hidden_sizes`` swaps the linear risk score for an MLP, which is the "deep
survival" of the specification. It is **off by default**, and the reason is
visible in the tests: on this data the linear model recovers the known ``theta``
and the MLP cannot be asked for a coefficient at all. Depth buys the ability to
represent interactions nobody has evidence for, at the cost of the one number
the decision layer needs to be able to explain. Turn it on when there are
covariates whose interactions you can actually validate.

Delta-CLV
---------

What L3 consumes::

    Delta-CLV(p) = sum_{k=1..H} [ S_k(p) - S_k(p_ref) ] * margin * (1 + d)^-k

the change in discounted expected future margin from pricing at ``p`` rather
than at the reference. It is **negative for a price rise** and positive for a
cut, which is exactly the counterweight the objective needs: the discount that
buys nothing today may still pay for itself, and the rise that prints margin
today may not.

Delta-CLV is a *modelled* quantity and inherits the discount rate, the horizon
and the survival specification. All three are declared configuration carrying
provenance records, and :func:`clv_sensitivity` sweeps them, because a single
number here would imply a precision the model does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from prismprice import config
from prismprice.compute import require_gpu

__all__ = [
    "CLVEstimate",
    "CoxSurvivalModel",
    "SurvivalFit",
    "clv_sensitivity",
    "price_shock",
]


def price_shock(
    price: NDArray[np.float64] | float, reference: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    """``p / p_ref - 1``: the quantity a customer actually reacts to.

    Zero when the price matches the reference, positive when it is above.
    """
    shock: NDArray[np.float64] = (
        np.asarray(price, dtype=float) / np.asarray(reference, dtype=float) - 1.0
    )
    return shock


@dataclass(frozen=True)
class SurvivalFit:
    """A fitted hazard model, with the evidence for trusting its coefficient."""

    feature_names: tuple[str, ...]
    coefficients: NDArray[np.float64]
    standard_errors: NDArray[np.float64]
    log_partial_likelihood: float
    n_events: int
    n_censored: int
    n_epochs: int
    converged: bool
    device: str
    is_linear: bool

    def coefficient_for(self, name: str) -> float:
        """The fitted log-hazard ratio on *name*.

        Raises:
            LookupError: if the risk score is a network, because a non-linear
                model has no single coefficient and returning one anyway is how
                a decision record ends up quoting a number that does not exist.
        """
        if not self.is_linear:
            raise LookupError(
                "the risk score is an MLP, so there is no coefficient to report; "
                "refit with hidden_sizes=() to obtain an interpretable theta"
            )
        return float(self.coefficients[self.feature_names.index(name)])

    def confidence_interval(self, name: str, z: float = 1.959963984540054) -> tuple[float, float]:
        """Wald interval on the coefficient, from the observed information."""
        index = self.feature_names.index(name)
        point = float(self.coefficients[index])
        error = float(self.standard_errors[index])
        return point - z * error, point + z * error

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "coefficients": [float(c) for c in self.coefficients],
            "standard_errors": [float(s) for s in self.standard_errors],
            "log_partial_likelihood": self.log_partial_likelihood,
            "n_events": self.n_events,
            "n_censored": self.n_censored,
            "converged": self.converged,
            "device": self.device,
            "is_linear": self.is_linear,
        }


@dataclass(frozen=True)
class CLVEstimate:
    """Change in discounted expected future margin from one pricing decision."""

    price: float
    reference_price: float
    delta_clv: float
    unit_margin: float
    horizon_periods: int
    discount_rate: float
    period_days: int
    survival_at_horizon_price: float
    survival_at_horizon_reference: float

    @property
    def price_shock(self) -> float:
        return self.price / self.reference_price - 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "price": self.price,
            "reference_price": self.reference_price,
            "price_shock": self.price_shock,
            "delta_clv": self.delta_clv,
            "unit_margin": self.unit_margin,
            "horizon_periods": self.horizon_periods,
            "discount_rate": self.discount_rate,
            "period_days": self.period_days,
        }


@dataclass
class CoxSurvivalModel:
    """Cox proportional-hazards repurchase model, fitted on GPU.

    Args:
        duration_column: Gap to the next purchase, in days.
        event_column: True when a repurchase was observed, False when the
            observation window ended first. Censoring is not missingness — a
            customer who has not yet returned still carries information that
            they survived this long, and dropping them biases every curve
            toward the customers who churned fastest.
        feature_columns: Risk-score inputs. Defaults to the price shock alone.
        cluster_column: Grouping for the robust variance. Repurchase data is
            *recurrent-event* data — one customer contributes many gaps — so the
            independence the partial likelihood assumes does not hold and naive
            standard errors are too narrow. ``None`` disables the correction.
        hidden_sizes: Empty for a linear risk score (interpretable, and the
            default); a tuple of widths for an MLP.
        period_days: Days per CLV period.
    """

    duration_column: str = "gap_days"
    event_column: str = "repurchased"
    feature_columns: tuple[str, ...] = ("price_shock",)
    cluster_column: str | None = "customer_id"
    hidden_sizes: tuple[int, ...] = ()
    period_days: int = 30
    learning_rate: float = 0.05
    max_epochs: int = 400
    tolerance: float = 1e-7
    l2_penalty: float = 0.0
    seed: int = config.DEFAULT_SEED

    fit_result: SurvivalFit | None = field(default=None, init=False)
    _module: Any = field(default=None, init=False, repr=False)
    _baseline_times: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0), init=False, repr=False
    )
    _baseline_cumulative_hazard: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0), init=False, repr=False
    )

    # -- fitting -----------------------------------------------------------

    def fit(self, purchases: pd.DataFrame) -> CoxSurvivalModel:
        """Fit by maximising the Breslow partial likelihood.

        Args:
            purchases: One row per purchase, carrying the gap to the next
                purchase, the event indicator, and the risk-score features.

        Returns:
            Self, with :attr:`fit_result` populated.

        Raises:
            ValueError: on missing columns or no observed events.
        """
        import torch

        missing = sorted(
            {self.duration_column, self.event_column, *self.feature_columns}
            - set(purchases.columns)
        )
        if missing:
            raise ValueError(f"purchases frame is missing columns: {missing}")

        durations = purchases[self.duration_column].to_numpy(dtype=float)
        events = purchases[self.event_column].to_numpy(dtype=bool)
        features = purchases[list(self.feature_columns)].to_numpy(dtype=float)
        clusters = (
            purchases[self.cluster_column].to_numpy()
            if self.cluster_column and self.cluster_column in purchases.columns
            else None
        )

        usable = np.isfinite(durations) & (durations > 0) & np.isfinite(features).all(axis=1)
        durations, events, features = durations[usable], events[usable], features[usable]
        if clusters is not None:
            clusters = clusters[usable]

        if not events.any():
            raise ValueError(
                "no observed repurchases: the partial likelihood is defined by the "
                "order of events, so a frame of only censored rows carries no signal"
            )

        device = require_gpu("estimation.retention")
        torch.manual_seed(self.seed)

        # Descending time puts the risk set at each row in a running prefix, so
        # the whole likelihood is one cumulative sum rather than a loop over
        # risk sets.
        order = np.argsort(-durations, kind="stable")
        durations, events, features = durations[order], events[order], features[order]
        if clusters is not None:
            clusters = clusters[order]

        x = torch.tensor(features, dtype=torch.float64, device=device)
        event_mask = torch.tensor(events, dtype=torch.bool, device=device)
        tied_index = torch.tensor(
            self._last_index_of_tied_group(durations), dtype=torch.long, device=device
        )

        self._module = self._build_module(x.shape[1], device)
        optimiser = torch.optim.Adam(
            self._module.parameters(), lr=self.learning_rate, weight_decay=self.l2_penalty
        )

        previous = float("inf")
        converged = False
        epochs_run = 0
        for step in range(1, self.max_epochs + 1):
            epochs_run = step
            optimiser.zero_grad()
            loss = -self._log_partial_likelihood(x, event_mask, tied_index)
            loss.backward()
            optimiser.step()

            current = float(loss.item())
            if abs(previous - current) < self.tolerance:
                converged = True
                break
            previous = current

        with torch.no_grad():
            final = float(self._log_partial_likelihood(x, event_mask, tied_index).item())

        coefficients, errors = self._coefficients_and_errors(
            x, event_mask, tied_index, device, features, events, clusters
        )
        self._estimate_baseline(durations, events, x)

        self.fit_result = SurvivalFit(
            feature_names=tuple(self.feature_columns),
            coefficients=coefficients,
            standard_errors=errors,
            log_partial_likelihood=final,
            n_events=int(events.sum()),
            n_censored=int((~events).sum()),
            n_epochs=epochs_run,
            converged=converged,
            device=str(device),
            is_linear=not self.hidden_sizes,
        )
        return self

    @staticmethod
    def _last_index_of_tied_group(durations_descending: NDArray[np.float64]) -> NDArray[np.int64]:
        """For each row, the index of the last row sharing its duration.

        The risk set at time ``t`` is every row with duration ``>= t``, so with
        durations sorted descending a prefix sum almost gives it — except that
        rows tied at ``t`` appear *after* the first of them and would be left
        out. Pointing each tied row at the last member of its group includes the
        whole tie, which is the Breslow handling. Gaps in this data are whole
        days, so ties are the common case rather than an edge case, and getting
        this wrong shrinks every risk set slightly and biases the coefficient.
        """
        n = durations_descending.shape[0]
        last = np.empty(n, dtype=np.int64)
        start = 0
        for i in range(1, n + 1):
            if i == n or durations_descending[i] != durations_descending[start]:
                last[start:i] = i - 1
                start = i
        return last

    def _build_module(self, n_features: int, device: Any) -> Any:
        import torch
        from torch import nn

        if not self.hidden_sizes:
            layer = nn.Linear(n_features, 1, bias=False, dtype=torch.float64)
            nn.init.zeros_(layer.weight)
            return layer.to(device)

        layers: list[Any] = []
        width = n_features
        for size in self.hidden_sizes:
            layers += [nn.Linear(width, size, dtype=torch.float64), nn.ReLU()]
            width = size
        layers.append(nn.Linear(width, 1, bias=False, dtype=torch.float64))
        return nn.Sequential(*layers).to(device)

    def _log_partial_likelihood(self, x: Any, event_mask: Any, tied_index: Any) -> Any:
        import torch

        eta = self._module(x).squeeze(-1)
        # Shift before exponentiating: risk scores of a few tens overflow float64
        # exp() and turn the whole likelihood into NaN, which reads as a
        # divergent fit rather than an arithmetic fault.
        shifted = eta - eta.max()
        cumulative = torch.cumsum(torch.exp(shifted), dim=0)
        risk_set = cumulative[tied_index]
        log_risk = torch.log(risk_set) + eta.max()
        return torch.sum((eta - log_risk)[event_mask])

    def _coefficients_and_errors(
        self,
        x: Any,
        event_mask: Any,
        tied_index: Any,
        device: Any,
        features: NDArray[np.float64],
        events: NDArray[np.bool_],
        clusters: NDArray[Any] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Coefficients plus Wald errors from the observed information matrix.

        The standard error is the square root of the diagonal of the inverse
        Hessian of the negative log partial likelihood — computed by autograd
        rather than derived by hand, which keeps it correct if the risk score
        changes.
        """
        import torch

        n_features = x.shape[1]
        if not self.hidden_sizes:
            weight = self._module.weight.detach().cpu().numpy().ravel()
        else:
            # No coefficient vector exists for a network; report NaNs rather
            # than something plausible-looking.
            return (np.full(n_features, np.nan), np.full(n_features, np.nan))

        def negative_log_likelihood(flat: Any) -> Any:
            eta = (x @ flat.reshape(n_features, 1)).squeeze(-1)
            shifted = eta - eta.max()
            cumulative = torch.cumsum(torch.exp(shifted), dim=0)
            log_risk = torch.log(cumulative[tied_index]) + eta.max()
            return -torch.sum((eta - log_risk)[event_mask])

        flat = torch.tensor(weight, dtype=torch.float64, device=device, requires_grad=True)
        hessian = torch.autograd.functional.hessian(  # type: ignore[no-untyped-call]
            negative_log_likelihood, flat
        )
        hessian = hessian.reshape(n_features, n_features).detach().cpu().numpy()

        try:
            information = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            return weight, np.full(n_features, np.nan)

        if clusters is None:
            errors = np.sqrt(np.clip(np.diag(information), 0.0, None))
            return weight, errors

        residuals = self._score_residuals(features, events, weight)
        covariance = self._clustered_covariance(residuals, clusters, information)
        return weight, np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    @staticmethod
    def _clustered_covariance(
        residuals: NDArray[np.float64],
        clusters: NDArray[Any],
        information: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Lin-Wei sandwich: ``I^-1 (sum_c U_c U_c') I^-1``.

        Score residuals are summed *within* a customer before being squared,
        which is what lets correlated repeat gaps count as one observation
        instead of many.
        """
        frame = pd.DataFrame(residuals)
        frame["__cluster"] = clusters
        grouped = frame.groupby("__cluster", sort=False).sum().to_numpy(dtype=float)
        meat = grouped.T @ grouped
        sandwich: NDArray[np.float64] = np.asarray(information @ meat @ information, dtype=float)
        return sandwich

    def _score_residuals(
        self,
        features: NDArray[np.float64],
        events: NDArray[np.bool_],
        weight: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Per-observation influence on the score, for the robust variance.

        For observation *i* with risk score ``w_i = exp(eta_i)``::

            u_i = d_i (x_i - xbar_i) - w_i * sum_{k: t_k <= T_i, d_k=1}
                                            (x_i - xbar_k) / S0_k

        The first term is the familiar "observed minus expected covariate at
        the event"; the second is what observation *i* contributes by sitting in
        everyone else's risk set, and dropping it (as a naive implementation
        does) understates the variance in exactly the direction that makes an
        interval look better than it is. Rows arrive sorted by descending
        duration, so both sums over ``t_k <= T_i`` are reverse cumulative sums.
        """
        eta = features @ weight
        w = np.exp(eta - eta.max())

        s0 = np.cumsum(w)
        s1 = np.cumsum(w[:, None] * features, axis=0)
        xbar = s1 / np.clip(s0, 1e-300, None)[:, None]

        event_indicator = events.astype(float)
        inverse_s0 = np.where(events, event_indicator / np.clip(s0, 1e-300, None), 0.0)
        xbar_over_s0 = (
            np.where(events, event_indicator, 0.0)[:, None]
            * xbar
            / np.clip(s0, 1e-300, None)[:, None]
        )

        # Reverse cumulative sums: index i needs every event at a duration at or
        # below its own, which is everything from i onwards once sorted
        # descending.
        tail_inverse_s0 = np.cumsum(inverse_s0[::-1])[::-1]
        tail_xbar_over_s0 = np.cumsum(xbar_over_s0[::-1], axis=0)[::-1]

        first = event_indicator[:, None] * (features - xbar)
        second = w[:, None] * (features * tail_inverse_s0[:, None] - tail_xbar_over_s0)
        residuals: NDArray[np.float64] = np.asarray(first - second, dtype=float)
        return residuals

    def _estimate_baseline(
        self, durations_descending: NDArray[np.float64], events: NDArray[np.bool_], x: Any
    ) -> None:
        """Breslow estimate of the cumulative baseline hazard.

        Cox gives the coefficient without the baseline; converting a hazard
        ratio into an absolute survival probability — which is what Delta-CLV
        needs — requires estimating it afterwards.
        """
        import torch

        with torch.no_grad():
            eta = self._module(x).squeeze(-1).cpu().numpy()

        exp_eta = np.exp(eta - eta.max())
        risk_denominator = np.cumsum(exp_eta)
        last = self._last_index_of_tied_group(durations_descending)

        times: list[float] = []
        increments: list[float] = []
        n = durations_descending.shape[0]
        i = 0
        while i < n:
            group_end = int(last[i])
            group = slice(i, group_end + 1)
            deaths = int(events[group].sum())
            if deaths:
                denominator = float(risk_denominator[group_end]) * float(np.exp(eta.max()))
                times.append(float(durations_descending[i]))
                increments.append(deaths / max(denominator, 1e-300))
            i = group_end + 1

        # Built descending; flip so the cumulative hazard is increasing in time.
        order = np.argsort(np.asarray(times, dtype=float))
        self._baseline_times = np.asarray(times, dtype=float)[order]
        self._baseline_cumulative_hazard = np.cumsum(np.asarray(increments, dtype=float)[order])

    # -- prediction --------------------------------------------------------

    def risk_score(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Linear predictor ``eta`` for each row of *features*."""
        import torch

        self._require_fit()
        with torch.no_grad():
            tensor = torch.tensor(
                np.atleast_2d(features), dtype=torch.float64, device=self._device()
            )
            out: NDArray[np.float64] = self._module(tensor).squeeze(-1).cpu().numpy()
        return out

    def survival(self, features: NDArray[np.float64], days: NDArray[np.float64]) -> Any:
        """``S(t | x)``: probability the customer has **not yet returned** by *t*.

        Uses ``S(t|x) = S_0(t) ** exp(eta)``, the proportional-hazards identity.

        **Read the direction carefully.** The event being modelled is a
        repurchase, so this is the survival function of the *inter-purchase gap*
        and high values mean the customer is taking a long time to come back —
        which is bad retention, the opposite of what "survival" suggests. The
        quantity that means "still a customer" is :meth:`repurchase_probability`.
        Delta-CLV is built on that one, and an early version of this module built
        it on this one: the sign came out inverted, so a price rise appeared to
        *improve* customer value and the objective would have been rewarded for
        raising prices. The tests that caught it are
        ``test_delta_clv_is_negative_for_a_price_rise`` and its neighbours.
        """
        self._require_fit()
        eta = self.risk_score(features)
        cumulative = np.interp(
            np.asarray(days, dtype=float),
            self._baseline_times,
            self._baseline_cumulative_hazard,
            left=0.0,
            right=(
                float(self._baseline_cumulative_hazard[-1])
                if self._baseline_cumulative_hazard.size
                else 0.0
            ),
        )
        return np.exp(-np.outer(np.exp(eta), cumulative))

    def repurchase_probability(
        self, features: NDArray[np.float64], days: NDArray[np.float64]
    ) -> Any:
        """``1 - S(t | x)``: probability the customer HAS returned by *t*.

        This is the retention quantity. It rises when the hazard rises, so a
        price cut (which raises the repurchase hazard) raises it, and a price
        rise lowers it.
        """
        return 1.0 - self.survival(features, days)

    def delta_clv(
        self,
        price: float,
        reference_price: float,
        unit_margin: float,
        horizon_periods: int = config.DEFAULT_CLV_HORIZON_PERIODS,
        discount_rate: float = config.DEFAULT_CLV_DISCOUNT_RATE,
        extra_features: dict[str, float] | None = None,
    ) -> CLVEstimate:
        """Change in discounted expected future margin from pricing at *price*.

        Negative for a price rise, positive for a cut. This is the term the L3
        objective weights by lambda.

        Args:
            price: Candidate price.
            reference_price: What the customer expects to pay.
            unit_margin: Contribution margin per retained period.
            horizon_periods: Periods to accumulate over.
            discount_rate: Per-period discount.
            extra_features: Values for risk features other than the price shock.

        Raises:
            ValueError: on a non-positive reference price or horizon.
        """
        self._require_fit()
        if reference_price <= 0:
            raise ValueError(f"reference_price must be > 0, got {reference_price}")
        if horizon_periods < 1:
            raise ValueError(f"horizon_periods must be >= 1, got {horizon_periods}")

        periods = np.arange(1, horizon_periods + 1, dtype=float)
        days = periods * self.period_days

        # Repurchase probability, not gap-survival. See survival()'s docstring:
        # using the latter inverts the sign of every Delta-CLV.
        at_price = self.repurchase_probability(
            self._features_for(price, reference_price, extra_features), days
        )[0]
        at_reference = self.repurchase_probability(
            self._features_for(reference_price, reference_price, extra_features), days
        )[0]

        discount = (1.0 + discount_rate) ** (-periods)
        delta = float(np.sum((at_price - at_reference) * unit_margin * discount))

        return CLVEstimate(
            price=price,
            reference_price=reference_price,
            delta_clv=delta,
            unit_margin=unit_margin,
            horizon_periods=horizon_periods,
            discount_rate=discount_rate,
            period_days=self.period_days,
            survival_at_horizon_price=float(at_price[-1]),
            survival_at_horizon_reference=float(at_reference[-1]),
        )

    def _features_for(
        self, price: float, reference_price: float, extra: dict[str, float] | None
    ) -> NDArray[np.float64]:
        values = []
        for name in self.feature_columns:
            if name == "price_shock":
                values.append(price / reference_price - 1.0)
            elif extra and name in extra:
                values.append(float(extra[name]))
            else:
                values.append(0.0)
        return np.asarray([values], dtype=float)

    def _device(self) -> Any:
        return next(self._module.parameters()).device

    def _require_fit(self) -> None:
        if self.fit_result is None or self._module is None:
            raise ValueError("fit() has not been called")


def clv_sensitivity(
    model: CoxSurvivalModel,
    price: float,
    reference_price: float,
    unit_margin: float,
    horizons: tuple[int, ...] = (6, 12, 24),
    discount_rates: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Delta-CLV across horizon and discount rate.

    Shipped because Delta-CLV is a modelled quantity whose two most arbitrary
    inputs are both configuration. A single number invites the reader to treat
    it as measured; a grid shows how much of it is the model and how much is the
    assumption, which is the difference between a finance input and a finding.
    """
    rows = []
    for horizon in horizons:
        for rate in discount_rates:
            estimate = model.delta_clv(
                price, reference_price, unit_margin, horizon_periods=horizon, discount_rate=rate
            )
            rows.append(
                {
                    "horizon_periods": horizon,
                    "discount_rate": rate,
                    "delta_clv": estimate.delta_clv,
                }
            )
    return pd.DataFrame(rows)
