"""
Cold-start prior tests (L1).

The blending logic is tested exhaustively with hand-built vectors; the encoder is
tested only for its GPU contract, since downloading a transformer in CI would be
slow and would not test anything this module is responsible for.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from prismprice import compute
from prismprice.compute import GPUInfo, GPUUnavailableError
from prismprice.features.embeddings import (
    ColdStartPrior,
    TransformerTextEncoder,
    knn_prior,
    normalise,
)


def basis(index: int, dim: int = 4) -> np.ndarray:
    vector = np.zeros(dim)
    vector[index] = 1.0
    return vector


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalise_gives_unit_rows():
    normalised = normalise(np.array([[3.0, 4.0], [1.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), [1.0, 1.0])


def test_normalise_leaves_zero_rows_as_zero_not_nan():
    """A zero embedding is a missing embedding; NaN would poison every similarity."""
    normalised = normalise(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert np.isfinite(normalised).all()
    np.testing.assert_allclose(normalised[0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# Neighbour blending
# ---------------------------------------------------------------------------


def test_identical_neighbour_dominates_the_blend():
    catalogue = np.array([basis(0), basis(1), basis(2)])
    prior = knn_prior(basis(0), catalogue, ["A", "B", "C"], [-2.0, -1.0, -0.5], k=3)
    assert prior.neighbours == ("A",)
    assert prior.beta == pytest.approx(-2.0)


def test_blend_is_similarity_weighted():
    target = np.array([1.0, 1.0, 0.0, 0.0])
    catalogue = np.array([basis(0), basis(1)])
    prior = knn_prior(target, catalogue, ["A", "B"], [-2.0, -1.0], k=2)
    # Equidistant neighbours weight equally.
    assert prior.beta == pytest.approx(-1.5)
    assert set(prior.neighbours) == {"A", "B"}


def test_distant_neighbours_are_dropped_not_downweighted():
    """A weighted mean over the whole catalogue is the catalogue mean in disguise."""
    catalogue = np.array([basis(0), basis(1), basis(2)])
    prior = knn_prior(basis(1), catalogue, ["A", "B", "C"], [-2.0, -1.0, -0.5], k=3)
    assert prior.neighbours == ("B",)


def test_no_neighbour_above_threshold_returns_an_unusable_prior():
    """Better to admit there is no analogue than to price on a fabricated one."""
    catalogue = np.array([basis(0)])
    prior = knn_prior(basis(1), catalogue, ["A"], [-2.0], k=1)
    assert not prior.is_usable
    assert prior.confidence == "low"
    assert prior.neighbours == ()


def test_zero_embedding_target_is_unusable():
    catalogue = np.array([basis(0), basis(1)])
    prior = knn_prior(np.zeros(4), catalogue, ["A", "B"], [-2.0, -1.0])
    assert not prior.is_usable


def test_k_caps_the_number_of_neighbours():
    catalogue = normalise(np.ones((10, 4)) + np.eye(10, 4) * 0.1)
    prior = knn_prior(np.ones(4), catalogue, [f"S{i}" for i in range(10)], [-1.5] * 10, k=3)
    assert len(prior.neighbours) == 3


def test_agreeing_neighbours_give_high_confidence_and_zero_spread():
    catalogue = normalise(np.ones((4, 4)) + np.eye(4) * 0.05)
    prior = knn_prior(np.ones(4), catalogue, list("ABCD"), [-1.8] * 4, k=4)
    assert prior.confidence == "high"
    assert prior.beta_sd == pytest.approx(0.0, abs=1e-9)


def test_disagreeing_neighbours_are_tagged_low_confidence():
    """A wide neighbourhood is a reason to distrust the prior, and must say so."""
    catalogue = normalise(np.ones((4, 4)) + np.eye(4) * 0.05)
    prior = knn_prior(np.ones(4), catalogue, list("ABCD"), [-0.2, -1.0, -2.5, -4.0], k=4)
    assert prior.confidence == "low"
    assert prior.beta_sd > 0.6


def test_similarities_are_reported_alongside_the_neighbours():
    catalogue = np.array([basis(0), basis(1)])
    prior = knn_prior(np.array([1.0, 0.5, 0.0, 0.0]), catalogue, ["A", "B"], [-2.0, -1.0], k=2)
    assert len(prior.similarities) == len(prior.neighbours)
    assert all(0.0 <= s <= 1.0 for s in prior.similarities)
    assert prior.similarities[0] >= prior.similarities[-1], "neighbours come back ranked"


def test_weights_sum_to_one_so_the_blend_stays_in_range():
    catalogue = normalise(np.ones((5, 4)) + np.eye(5, 4) * 0.2)
    betas = [-2.5, -2.0, -1.5, -1.0, -0.5]
    prior = knn_prior(np.ones(4), catalogue, list("ABCDE"), betas, k=5)
    assert min(betas) <= prior.beta <= max(betas)


def test_mismatched_catalogue_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        knn_prior(basis(0), np.array([basis(0), basis(1)]), ["A", "B"], [-1.0])


def test_embedding_row_count_must_match_the_sku_list():
    with pytest.raises(ValueError, match="align"):
        knn_prior(basis(0), np.array([basis(0)]), ["A", "B"], [-1.0, -2.0])


def test_prior_is_immutable():
    prior = ColdStartPrior(-1.5, 0.2, ("A",), (0.9,), "high")
    with pytest.raises(FrozenInstanceError):
        prior.beta = -2.0


# ---------------------------------------------------------------------------
# Encoder GPU contract
# ---------------------------------------------------------------------------


def test_encoder_requires_a_gpu(monkeypatch):
    """Encoding a catalogue is ML, so it obeys the same device contract."""
    monkeypatch.delenv("PRISMPRICE_ALLOW_CPU", raising=False)
    monkeypatch.setattr(compute, "gpu_report", lambda: GPUInfo(available=False, reason="simulated"))
    with pytest.raises(GPUUnavailableError, match=r"features\.embeddings"):
        TransformerTextEncoder().encode(["a red mug"])
