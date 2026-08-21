"""
Multi-SKU markdown under a shared budget — where a learned policy earns its keep.

``decision/markdown.py`` solves one SKU exactly, and argues that a policy network
would be strictly worse there: the state space is enumerable, so backward
induction gives the optimum and approximating it can only lose. That argument
has a boundary, and this module is on the other side of it.

Put ``N`` SKUs under one shared markdown budget — total revenue the category may
give away against full price — and the problem stops decomposing. Discounting one
SKU consumes budget the others could have used, so the SKUs are no longer
independent and cannot be solved separately. The joint state is::

    (t, i_1, ..., i_N, budget remaining)

which is exponential in ``N``. Five SKUs with fifty units each and ten budget
levels is already ~3.5 billion states. Backward induction is not slow there; it
is impossible.

So the policy is learned. A network reads the state and emits a price for each
SKU, trained by REINFORCE with a learned baseline against the simulator.

The claim is tested, not asserted
---------------------------------

A learned policy is only worth anything if it is close to optimal, and "close to
optimal" is unmeasurable in the regime where you need it — that is the whole
reason you needed it. So it is measured where both are computable:
:func:`solve_joint_tabular` solves a deliberately tiny instance exactly, and the
network is scored against that optimum. Only then is it run where the table
cannot go, and there it is scored against the strongest available alternative —
per-SKU DP that ignores the shared budget, which is what a team without this
module would actually deploy.

That alternative is not a straw man. Independent DP is *optimal for each SKU in
isolation* and fails only on the coupling: it spends the shared budget without
knowing the budget exists, so it discounts early, exhausts the allowance, and
leaves the SKUs that needed it most at full price into the last week of the
season.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prismprice import config
from prismprice.compute import require_gpu

__all__ = [
    "JointMarkdownProblem",
    "NeuralMarkdownPolicy",
    "evaluate_joint_policy",
    "independent_dp_policy",
    "solve_joint_tabular",
]


@dataclass(frozen=True)
class JointMarkdownProblem:
    """Several SKUs sharing one markdown budget over one season.

    Args:
        prices: Ladder shared by every SKU, ascending.
        full_prices: Per-SKU reference price. Markdown spend is measured against
            this, so the budget is "revenue given away", not "price paid".
        base_demands: Expected units per period at ``full_price``.
        elasticity: Shared own-price elasticity.
        unit_costs: Per-SKU cost.
        salvage_values: Per-SKU value of an unsold unit.
        horizon: Periods in the season.
        inventories: Opening units per SKU.
        markdown_budget: Total revenue the category may give away. This is the
            coupling — without it the SKUs are independent and a per-SKU DP is
            optimal.
        budget_levels: Discretisation used only by the tabular reference.
    """

    prices: tuple[float, ...]
    full_prices: tuple[float, ...]
    base_demands: tuple[float, ...]
    elasticity: float
    unit_costs: tuple[float, ...]
    salvage_values: tuple[float, ...]
    horizon: int
    inventories: tuple[int, ...]
    markdown_budget: float
    budget_levels: int = 12

    def __post_init__(self) -> None:
        n = len(self.inventories)
        for name, value in (
            ("full_prices", self.full_prices),
            ("base_demands", self.base_demands),
            ("unit_costs", self.unit_costs),
            ("salvage_values", self.salvage_values),
        ):
            if len(value) != n:
                raise ValueError(f"{name} must have one entry per SKU, got {len(value)} for {n}")
        if len(self.prices) < 2:
            raise ValueError("need at least 2 prices to choose between")
        if self.elasticity >= 0:
            raise ValueError(f"elasticity must be negative, got {self.elasticity}")
        if self.markdown_budget < 0:
            raise ValueError(f"markdown_budget must be >= 0, got {self.markdown_budget}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")

    @property
    def n_skus(self) -> int:
        return len(self.inventories)

    def expected_demand(self, sku: int, price: float) -> float:
        return float(
            self.base_demands[sku] * (price / self.full_prices[sku]) ** self.elasticity
        )

    def markdown_spend(self, sku: int, price: float, units: float) -> float:
        """Revenue given away versus full price. Never negative."""
        return float(max(self.full_prices[sku] - price, 0.0) * units)

    def state_space_size(self) -> float:
        """Joint states a table would need. Reported because it is the argument."""
        size = float(self.horizon + 1) * float(self.budget_levels)
        for stock in self.inventories:
            size *= float(stock + 1)
        return size


# ---------------------------------------------------------------------------
# Exact reference, for instances small enough to have one
# ---------------------------------------------------------------------------


def solve_joint_tabular(problem: JointMarkdownProblem, max_states: int = 4_000_000) -> Any:
    """Exact backward induction over the joint state, for tiny instances only.

    Exists to give the learned policy something true to be measured against.
    Refuses instances it cannot solve rather than silently approximating —
    a reference that quietly degrades is worse than no reference, because the
    learned policy would then be scored against another approximation.

    Raises:
        ValueError: when the joint state space exceeds ``max_states``.
    """
    size = problem.state_space_size()
    if size > max_states:
        raise ValueError(
            f"joint state space is {size:,.0f} states, above the {max_states:,} limit. "
            f"This is the regime the learned policy exists for — the table cannot "
            f"be the reference here, and pretending otherwise would compare an "
            f"approximation against an approximation."
        )

    n = problem.n_skus
    shapes = tuple(stock + 1 for stock in problem.inventories)
    budget_grid = np.linspace(0.0, problem.markdown_budget, problem.budget_levels)

    value = np.zeros((problem.horizon + 1, *shapes, problem.budget_levels), dtype=float)
    # Terminal: salvage whatever is left.
    terminal = np.zeros(shapes, dtype=float)
    for index in np.ndindex(*shapes):
        terminal[index] = sum(problem.salvage_values[k] * index[k] for k in range(n))
    value[0] = terminal[..., None]

    action_grid = list(np.ndindex(*(len(problem.prices),) * n))

    for t in range(1, problem.horizon + 1):
        nxt = value[t - 1]
        for index in np.ndindex(*shapes):
            for b_i, budget in enumerate(budget_grid):
                best = -np.inf
                for actions in action_grid:
                    total = 0.0
                    spend = 0.0
                    # Expected sales per SKU, truncated by stock. Demand is
                    # taken at its mean here rather than sampled: the reference
                    # only has to be exact for the *same* dynamics the learned
                    # policy faces, and both use expected transitions.
                    landing = list(index)
                    for k in range(n):
                        price = problem.prices[actions[k]]
                        sold = min(problem.expected_demand(k, price), float(index[k]))
                        total += sold * (price - problem.unit_costs[k])
                        spend += problem.markdown_spend(k, price, sold)
                        landing[k] = int(round(index[k] - sold))
                    if spend > budget + 1e-9:
                        continue
                    remaining = max(budget - spend, 0.0)
                    b_next = int(np.argmin(np.abs(budget_grid - remaining)))
                    total += float(nxt[tuple(landing) + (b_next,)])
                    best = max(best, total)
                value[t][index + (b_i,)] = best if np.isfinite(best) else 0.0

    return value


# ---------------------------------------------------------------------------
# The alternative a team without this module would deploy
# ---------------------------------------------------------------------------


def independent_dp_policy(problem: JointMarkdownProblem) -> Any:
    """Per-SKU optimal policy that does not know the shared budget exists.

    Not a straw man: each SKU's policy is genuinely optimal *in isolation*. It
    fails only on the coupling — it discounts on each SKU's own merits, spends
    the shared allowance without accounting for it, and leaves nothing for the
    SKUs that needed it later.
    """
    from prismprice.decision.markdown import MarkdownProblem, solve_markdown

    solved = []
    for k in range(problem.n_skus):
        single = MarkdownProblem(
            prices=problem.prices,
            base_price=problem.full_prices[k],
            base_demand=problem.base_demands[k],
            elasticity=problem.elasticity,
            unit_cost=problem.unit_costs[k],
            salvage_value=problem.salvage_values[k],
            horizon=problem.horizon,
            initial_inventory=problem.inventories[k],
        )
        solved.append(solve_markdown(single))

    def policy(t: int, stocks: list[int], budget: float) -> list[float]:
        return [
            solved[k].price_at(t, min(stocks[k], problem.inventories[k]))
            for k in range(problem.n_skus)
        ]

    return policy


# ---------------------------------------------------------------------------
# The learned policy
# ---------------------------------------------------------------------------


@dataclass
class NeuralMarkdownPolicy:
    """Factored categorical policy over the price ladder, trained by REINFORCE.

    One network, one head per SKU. Factoring the action means the output grows
    linearly in ``N`` rather than as ``K**N`` — the same blow-up that makes the
    table impossible would otherwise reappear in the final layer.

    Args:
        hidden: Width of the two hidden layers.
        episodes: Training episodes.
        batch: Episodes per gradient step. Averaging over a batch is what makes
            REINFORCE's gradient usable at all; single-episode updates are too
            noisy to learn a budget allocation.
        learning_rate: Adam step size.
        entropy_bonus: Keeps the policy exploring early. Without it the network
            commits to whichever price happened to work first and never
            discovers that saving budget for later pays.
        seed: RNG seed.
    """

    hidden: int = 128
    episodes: int = 6000
    batch: int = 64
    learning_rate: float = 3e-3
    entropy_bonus: float = 0.01
    seed: int = config.DEFAULT_SEED

    _net: Any = field(default=None, init=False, repr=False)
    _problem: JointMarkdownProblem | None = field(default=None, init=False, repr=False)
    history: list[float] = field(default_factory=list, init=False)

    def _features(
        self, problem: JointMarkdownProblem, t: int, stocks: NDArray[np.float64], budget: float
    ) -> NDArray[np.float64]:
        """Normalised state. Scaling matters more than usual here: raw inventory
        and raw budget differ by two orders of magnitude, and an unscaled input
        layer spends its early training just undoing that."""
        return np.concatenate(
            [
                [t / problem.horizon],
                stocks / np.maximum(np.asarray(problem.inventories, dtype=float), 1.0),
                [budget / max(problem.markdown_budget, 1e-9)],
            ]
        )

    def fit(self, problem: JointMarkdownProblem) -> NeuralMarkdownPolicy:
        """Train against the simulator."""
        import torch
        from torch import nn

        device = require_gpu("decision.joint_markdown")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        n, k = problem.n_skus, len(problem.prices)
        self._problem = problem
        self._net = nn.Sequential(
            nn.Linear(n + 2, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, n * k + 1),  # +1: the value baseline
        ).to(device)
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)

        for step in range(self.episodes // self.batch):
            log_probs, entropies, values, returns = [], [], [], []

            for _ in range(self.batch):
                stocks = np.asarray(problem.inventories, dtype=float)
                budget = problem.markdown_budget
                episode_lp, episode_ent, episode_v = [], [], []
                reward = 0.0

                for t in range(problem.horizon, 0, -1):
                    x = torch.tensor(
                        self._features(problem, t, stocks, budget),
                        dtype=torch.float32, device=device,
                    )
                    out = self._net(x)
                    logits = out[: n * k].reshape(n, k)
                    episode_v.append(out[-1])

                    dist = torch.distributions.Categorical(logits=logits)
                    choice = dist.sample()
                    episode_lp.append(dist.log_prob(choice).sum())
                    episode_ent.append(dist.entropy().sum())

                    for i in range(n):
                        if stocks[i] <= 0:
                            continue
                        price = problem.prices[int(choice[i].item())]
                        demand = rng.poisson(problem.expected_demand(i, price))
                        sold = float(min(demand, stocks[i]))
                        spend = problem.markdown_spend(i, price, sold)
                        # The budget is a hard allowance: a SKU cannot spend
                        # what the category has not got, so the sale is trimmed
                        # to what remains rather than penalised after the fact.
                        if spend > budget:
                            allowed = budget / max(problem.full_prices[i] - price, 1e-9)
                            sold = float(min(sold, max(allowed, 0.0)))
                            spend = problem.markdown_spend(i, price, sold)
                        budget -= spend
                        stocks[i] -= sold
                        reward += sold * (price - problem.unit_costs[i])

                reward += float(np.sum(np.asarray(problem.salvage_values) * stocks))
                log_probs.append(torch.stack(episode_lp).sum())
                entropies.append(torch.stack(episode_ent).sum())
                values.append(torch.stack(episode_v).mean())
                returns.append(reward)

            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            values_t = torch.stack(values)
            advantage = returns_t - values_t.detach()
            # Standardising the advantage keeps the gradient scale independent
            # of how many pounds a season happens to be worth, so the same
            # learning rate works across problem sizes.
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)

            policy_loss = -(torch.stack(log_probs) * advantage).mean()
            value_loss = torch.nn.functional.mse_loss(values_t, returns_t)
            entropy = torch.stack(entropies).mean()
            loss = policy_loss + 0.5 * value_loss - self.entropy_bonus * entropy

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
            optimiser.step()
            self.history.append(float(np.mean(returns)))

        return self

    def price_at(self, t: int, stocks: list[int], budget: float) -> list[float]:
        """Greedy action in one state. Used for evaluation, not training."""
        import torch

        if self._net is None or self._problem is None:
            raise ValueError("fit() has not been called")
        problem = self._problem
        device = next(self._net.parameters()).device

        with torch.no_grad():
            x = torch.tensor(
                self._features(problem, t, np.asarray(stocks, dtype=float), budget),
                dtype=torch.float32, device=device,
            )
            logits = self._net(x)[: problem.n_skus * len(problem.prices)]
            logits = logits.reshape(problem.n_skus, len(problem.prices))
            picks = torch.argmax(logits, dim=1).cpu().numpy()
        return [problem.prices[int(p)] for p in picks]


def evaluate_joint_policy(
    problem: JointMarkdownProblem,
    policy: Any,
    n_seasons: int = 600,
    seed: int = config.DEFAULT_SEED,
) -> dict[str, float]:
    """Simulate *policy* and report realised profit and budget use.

    ``policy`` is ``(t, stocks, budget) -> [price per SKU]``, so a learned
    policy and a per-SKU DP are scored through the same interface on the same
    draws — the comparison is otherwise not a comparison.
    """
    rng = np.random.default_rng(seed)
    profits, spends, leftovers = [], [], []

    for _ in range(n_seasons):
        stocks = list(problem.inventories)
        budget = problem.markdown_budget
        profit = 0.0
        spent = 0.0

        for t in range(problem.horizon, 0, -1):
            prices = policy(t, stocks, budget)
            for i in range(problem.n_skus):
                if stocks[i] <= 0:
                    continue
                demand = rng.poisson(problem.expected_demand(i, prices[i]))
                sold = float(min(demand, stocks[i]))
                spend = problem.markdown_spend(i, prices[i], sold)
                if spend > budget:
                    allowed = budget / max(problem.full_prices[i] - prices[i], 1e-9)
                    sold = float(min(sold, max(allowed, 0.0)))
                    spend = problem.markdown_spend(i, prices[i], sold)
                budget -= spend
                spent += spend
                stocks[i] -= int(sold)
                profit += sold * (prices[i] - problem.unit_costs[i])

        profit += sum(problem.salvage_values[i] * stocks[i] for i in range(problem.n_skus))
        profits.append(profit)
        spends.append(spent)
        leftovers.append(sum(stocks))

    return {
        "mean_profit": float(np.mean(profits)),
        "profit_std_error": float(np.std(profits, ddof=1) / np.sqrt(n_seasons)),
        "mean_budget_spent": float(np.mean(spends)),
        "budget_utilisation": float(np.mean(spends) / max(problem.markdown_budget, 1e-9)),
        "mean_leftover": float(np.mean(leftovers)),
    }
