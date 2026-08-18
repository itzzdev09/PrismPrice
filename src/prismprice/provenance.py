"""
Parameter provenance.

Every number that can change a recommended price must say where it came from.
This module makes that structural: a constant in the decision path without a
provenance record fails :func:`require_sourced_decision_path`, which CI runs.

The motivating problem is that three very different kinds of number were
previously indistinguishable in ``config.py`` — they were all bare floats:

* an **estimate** (elasticity, hazard) which must come from data and must never
  be hand-set;
* a **policy** dial (how much future customer value is worth against cash margin
  today) which *cannot* be estimated from data at all, because it encodes a
  preference rather than a fact;
* a **placeholder** that someone typed to make a test pass.

Collapsing them loses the only distinction that matters when a reviewer asks
"why 0.15?". A policy dial answers "because the category owner accepts this
trade"; a placeholder has no answer, and the point of this module is that it is
no longer allowed to look like it does.

Provenance kinds
----------------

``MEASURED``
    Estimated from data by a fitted model. Rejected in configuration on
    purpose: estimates belong in model artefacts, and a hand-written
    "measured" constant is a placeholder with better branding.
``LITERATURE``
    Taken from published research. Requires a citation specific enough to check.
``POLICY``
    A business preference. Requires an owner, an ``elicitation`` string stating
    the exact trade the number encodes, and a sensitivity bracket.
``TECHNICAL``
    Forced by a technical constraint (floating-point representation, protocol
    limits). The constraint goes in ``rationale``.
``PLACEHOLDER``
    Unsourced. Permitted outside the decision path; an error inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "PLACEHOLDER_IN_DECISION_PATH",
    "Parameter",
    "Provenance",
    "ProvenanceError",
    "audit_decision_path",
    "describe",
    "frontier",
    "gamma_from_tail_tradeoff",
    "iter_unsourced",
    "lambda_from_tradeoff",
    "register",
    "registry",
    "require_sourced_decision_path",
]


class Provenance(Enum):
    """Where a parameter's value came from. See module docstring."""

    MEASURED = "MEASURED"
    LITERATURE = "LITERATURE"
    POLICY = "POLICY"
    TECHNICAL = "TECHNICAL"
    PLACEHOLDER = "PLACEHOLDER"


class ProvenanceError(ValueError):
    """A provenance record is incomplete, self-contradictory, or missing."""


#: Reason code emitted when the audit finds an unsourced decision-path constant.
PLACEHOLDER_IN_DECISION_PATH = "PP-P001"


@dataclass(frozen=True)
class Parameter:
    """One configuration constant plus the evidence for its value.

    Args:
        name: The exported constant name, e.g. ``"DEFAULT_MARGIN_FLOOR_PCT"``.
        value: The value itself. ``config`` exports what :func:`register`
            returns, so the record and the number cannot drift apart.
        provenance: Which kind of evidence backs it.
        rationale: Why this value rather than a neighbouring one. Always
            required — a citation says where a number came from, not why it was
            chosen over the alternatives in the same paper.
        citation: Required for ``LITERATURE``. Specific enough to look up.
        owner: Required for ``POLICY``. Who decides this trade.
        elicitation: Required for ``POLICY``. The trade the number encodes,
            stated so it can be disagreed with directly.
        sensitivity: ``(low, high)`` bracket to sweep before the value is
            trusted. Required for ``POLICY``, because a preference shipped as a
            point estimate hides whether the recommendation flips inside the
            plausible range.
        in_decision_path: True when this value can change a recommended price.
        requires_local_elicitation: True when the shipped default is a
            defensible starting point rather than *this operator's* answer.
    """

    name: str
    value: Any
    provenance: Provenance
    rationale: str
    citation: str | None = None
    owner: str | None = None
    elicitation: str | None = None
    sensitivity: tuple[float, float] | None = None
    in_decision_path: bool = True
    requires_local_elicitation: bool = False

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise ProvenanceError(f"{self.name}: rationale is required for every parameter")

        if self.provenance is Provenance.MEASURED:
            raise ProvenanceError(
                f"{self.name}: MEASURED values are estimates and belong in model artefacts, "
                f"not in configuration. A hand-written measured constant is a placeholder."
            )

        if self.provenance is Provenance.LITERATURE and not (self.citation or "").strip():
            raise ProvenanceError(
                f"{self.name}: provenance is LITERATURE but no citation was given. "
                f"A literature value that cannot be looked up is a placeholder."
            )

        if self.provenance is Provenance.POLICY:
            if not (self.owner or "").strip():
                raise ProvenanceError(
                    f"{self.name}: provenance is POLICY but no owner was given. "
                    f"A preference with no owner is nobody's preference."
                )
            if not (self.elicitation or "").strip():
                raise ProvenanceError(
                    f"{self.name}: provenance is POLICY but no elicitation was given. "
                    f"State the trade the number encodes so it can be disagreed with."
                )
            if self.sensitivity is None:
                raise ProvenanceError(
                    f"{self.name}: provenance is POLICY but no sensitivity bracket was given. "
                    f"A preference shipped as a point estimate hides whether the "
                    f"recommendation flips inside the plausible range."
                )

        if self.sensitivity is not None:
            low, high = self.sensitivity
            if low > high:
                raise ProvenanceError(
                    f"{self.name}: sensitivity bracket {self.sensitivity} is inverted"
                )
            if isinstance(self.value, (int, float)) and not low <= float(self.value) <= high:
                raise ProvenanceError(
                    f"{self.name}: value {self.value} lies outside its own sensitivity "
                    f"bracket {self.sensitivity}"
                )

    @property
    def is_sourced(self) -> bool:
        """True when the value has evidence of any kind behind it."""
        return self.provenance is not Provenance.PLACEHOLDER

    def as_dict(self) -> dict[str, Any]:
        """Serialisable record, for attachment to a decision log."""
        return {
            "name": self.name,
            "value": self.value,
            "provenance": self.provenance.value,
            "rationale": self.rationale,
            "citation": self.citation,
            "owner": self.owner,
            "elicitation": self.elicitation,
            "sensitivity": list(self.sensitivity) if self.sensitivity else None,
            "in_decision_path": self.in_decision_path,
            "requires_local_elicitation": self.requires_local_elicitation,
        }


_REGISTRY: dict[str, Parameter] = {}


def register(param: Parameter) -> Any:
    """Record *param* and return its value.

    Returning the value is what keeps the record honest. ``config`` writes::

        X: Final[float] = register(Parameter(name="X", value=0.15, ...))

    so the number appears exactly once and cannot drift from its record.

    Raises:
        ProvenanceError: on a duplicate name.
    """
    if param.name in _REGISTRY:
        raise ProvenanceError(f"{param.name} is already registered")
    _REGISTRY[param.name] = param
    return param.value


def registry() -> dict[str, Parameter]:
    """Every registered parameter, by name."""
    return dict(_REGISTRY)


def iter_unsourced() -> Iterator[Parameter]:
    """Every unsourced parameter, decision path or not. For reporting."""
    yield from (p for p in _REGISTRY.values() if not p.is_sourced)


def audit_decision_path() -> list[Parameter]:
    """Return every decision-path parameter lacking a source.

    An empty list is the passing state.
    """
    return sorted(
        (p for p in _REGISTRY.values() if p.in_decision_path and not p.is_sourced),
        key=lambda p: p.name,
    )


def require_sourced_decision_path() -> None:
    """Raise if any decision-path constant is unsourced.

    Raises:
        ProvenanceError: listing every offender at once, so one run diagnoses
            the whole configuration rather than the first fault found.
    """
    unsourced = audit_decision_path()
    if unsourced:
        names = ", ".join(p.name for p in unsourced)
        raise ProvenanceError(
            f"{PLACEHOLDER_IN_DECISION_PATH}: {len(unsourced)} decision-path parameter(s) "
            f"have no source: {names}. Give each a provenance record, or mark it "
            f"in_decision_path=False if it cannot change a price."
        )


# ---------------------------------------------------------------------------
# Inverse specification of the objective weights
# ---------------------------------------------------------------------------
#
# lambda and gamma are neither estimated nor guessed. They are solved for from a
# trade the operator states in units they actually hold an opinion about.


def lambda_from_tradeoff(margin_sacrificed: float, clv_gained: float) -> float:
    r"""Solve for :math:`\lambda` in the L3 objective from a stated trade.

    The objective term is
    :math:`\mathbb{E}[q](p - c - \nu) + \lambda\,\mathbb{E}[\Delta\text{CLV}]`,
    so :math:`\lambda` is the exchange rate between one currency unit of
    contribution margin banked today and one currency unit of *modelled* future
    customer value. Indifference between the two sides gives
    :math:`\lambda = \text{margin sacrificed} / \Delta\text{CLV gained}`.

    Stating it this way is the whole point. Nobody has a calibrated intuition
    for "lambda = 0.30", but a category owner can answer "how much margin today
    would you give up for one pound of modelled future value?" — and an answer
    well below 1.0 is itself information: it prices trust in the CLV model, not
    only commercial strategy.

    Args:
        margin_sacrificed: Contribution margin per unit the operator will give
            up. Must be non-negative.
        clv_gained: Modelled ``ΔCLV`` per unit obtained in exchange. Must be
            strictly positive.

    Returns:
        The implied :math:`\lambda`.

    Raises:
        ValueError: on non-positive ``clv_gained`` or negative sacrifice.
    """
    if clv_gained <= 0:
        raise ValueError(
            f"clv_gained must be > 0 to define an exchange rate, got {clv_gained}. "
            f"A trade that gains no future value does not identify lambda."
        )
    if margin_sacrificed < 0:
        raise ValueError(f"margin_sacrificed must be >= 0, got {margin_sacrificed}")
    return margin_sacrificed / clv_gained


def gamma_from_tail_tradeoff(expected_sacrificed: float, cvar_reduced: float) -> float:
    r"""Solve for :math:`\gamma`, the CVaR penalty weight, from a stated trade.

    The objective subtracts :math:`\gamma \cdot \text{CVaR}_\alpha`, so
    :math:`\gamma` is how much *expected* contribution the operator will give up
    to remove one unit of expected loss in the worst :math:`\alpha` tail.
    :math:`\gamma = 0` is risk neutral.

    Args:
        expected_sacrificed: Expected contribution given up. Non-negative.
        cvar_reduced: Reduction in tail loss obtained. Must be > 0.

    Returns:
        The implied :math:`\gamma`.

    Raises:
        ValueError: on non-positive ``cvar_reduced`` or negative sacrifice.
    """
    if cvar_reduced <= 0:
        raise ValueError(f"cvar_reduced must be > 0 to define an exchange rate, got {cvar_reduced}")
    if expected_sacrificed < 0:
        raise ValueError(f"expected_sacrificed must be >= 0, got {expected_sacrificed}")
    return expected_sacrificed / cvar_reduced


def frontier(
    param: Parameter,
    score: Callable[[float], Any],
    steps: int = 11,
) -> list[tuple[float, Any]]:
    """Sweep *param* across its sensitivity bracket and score each value.

    This is what a policy dial needs before it is trusted. If the outcome is
    identical across the bracket the dial is not load-bearing and the exact
    value does not matter; if it flips, the value must be elicited locally
    rather than defaulted. ``score`` is supplied by the caller — L3 will pass
    its objective — so this is usable before the decision engine exists.

    Args:
        param: A parameter carrying a ``sensitivity`` bracket.
        score: Callable mapping a candidate value to any comparable outcome.
        steps: Number of points across the bracket, at least 2.

    Returns:
        ``(value, score)`` pairs in ascending value order.

    Raises:
        ValueError: if *param* has no sensitivity bracket, or ``steps < 2``.
    """
    if param.sensitivity is None:
        raise ValueError(f"{param.name} has no sensitivity bracket, so there is nothing to sweep.")
    if steps < 2:
        raise ValueError(f"steps must be >= 2 to span a bracket, got {steps}")

    low, high = param.sensitivity
    width = high - low
    values = [low + width * i / (steps - 1) for i in range(steps)]
    return [(v, score(v)) for v in values]


def describe(names: Sequence[str] | None = None) -> str:
    """Human-readable provenance table, for attaching to a run log."""
    params = (
        [_REGISTRY[n] for n in names]
        if names is not None
        else sorted(_REGISTRY.values(), key=lambda p: p.name)
    )
    lines = ["Parameter provenance", "=" * 64]
    for p in params:
        flag = "" if p.is_sourced else "   <-- UNSOURCED"
        lines.append(f"{p.name} = {p.value}  [{p.provenance.value}]{flag}")
        lines.append(f"    why:   {p.rationale}")
        if p.citation:
            lines.append(f"    cite:  {p.citation}")
        if p.owner:
            lines.append(f"    owner: {p.owner}")
        if p.elicitation:
            lines.append(f"    trade: {p.elicitation}")
        if p.sensitivity:
            lines.append(f"    sweep: {p.sensitivity[0]} .. {p.sensitivity[1]}")
        if p.requires_local_elicitation:
            lines.append("    NOTE:  repo default; re-elicit before production use")
    return "\n".join(lines)
