"""
Cold-start priors from product embeddings (L1).

A new SKU has no demand history, so :class:`~prismprice.features.builder.FeatureBuilder`
raises rather than inventing one. This module supplies the alternative: place the
new product in an embedding space built from its title, category and image, and
borrow an elasticity prior from its nearest historical neighbours.

Two parts, deliberately separated:

* :func:`knn_prior` — pure numerical blending. No model, no GPU, fully testable.
* :class:`TransformerTextEncoder` — the actual text/image encoder. This is ML,
  so it runs on GPU and raises rather than falling back (see README §8.1).

Keeping them apart means the borrowing logic — the part with the statistical
subtleties — can be tested exhaustively without a GPU or a model download.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from prismprice.compute import require_gpu

__all__ = [
    "ColdStartPrior",
    "EmbeddingEncoder",
    "TransformerTextEncoder",
    "knn_prior",
    "normalise",
]


def normalise(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """L2-normalise row-wise, leaving zero rows as zero rather than NaN."""
    array = np.atleast_2d(np.asarray(vectors, dtype=float))
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    result: NDArray[np.float64] = np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)
    return result


@dataclass(frozen=True)
class ColdStartPrior:
    """An elasticity prior borrowed from neighbours, with its own uncertainty."""

    beta: float
    """Similarity-weighted elasticity of the K nearest historical SKUs."""
    beta_sd: float
    """Weighted spread across those neighbours. Wide means the neighbourhood
    disagrees, which is a reason to distrust the prior — not to ignore it."""
    neighbours: tuple[str, ...]
    similarities: tuple[float, ...]
    confidence: str
    """``high`` | ``low``. Low when neighbours are distant or disagree."""

    @property
    def is_usable(self) -> bool:
        return bool(self.neighbours)


def knn_prior(
    target_embedding: NDArray[np.float64],
    catalogue_embeddings: NDArray[np.float64],
    catalogue_skus: Sequence[str],
    catalogue_betas: Sequence[float],
    k: int = 5,
    min_similarity: float = 0.30,
    max_beta_sd: float = 0.6,
) -> ColdStartPrior:
    """Blend neighbour elasticities by cosine similarity.

    Args:
        target_embedding: Embedding of the new SKU.
        catalogue_embeddings: ``(n, d)`` embeddings of SKUs with known elasticity.
        catalogue_skus: Identifiers aligned to ``catalogue_embeddings``.
        catalogue_betas: Estimated elasticities aligned to the same rows.
        k: Neighbours to blend.
        min_similarity: Neighbours below this are dropped entirely rather than
            down-weighted. A weighted average over the whole catalogue always
            returns *something*, and that something is the catalogue mean wearing
            a similarity score — worse than admitting there is no neighbour.
        max_beta_sd: Spread above which the prior is tagged ``low`` confidence.

    Returns:
        A :class:`ColdStartPrior`. With no neighbour above the threshold it comes
        back empty and ``is_usable`` is False; the caller degrades rather than
        pricing on a fabricated elasticity.
    """
    if len(catalogue_skus) != len(catalogue_betas):
        raise ValueError("catalogue_skus and catalogue_betas must be the same length")

    catalogue = normalise(catalogue_embeddings)
    if catalogue.shape[0] != len(catalogue_skus):
        raise ValueError("catalogue_embeddings rows must align with catalogue_skus")

    target = normalise(target_embedding)[0]
    if not np.any(target):
        return ColdStartPrior(0.0, 0.0, (), (), "low")

    similarities = catalogue @ target
    order = np.argsort(similarities)[::-1][:k]
    keep = [i for i in order if similarities[i] >= min_similarity]

    if not keep:
        return ColdStartPrior(0.0, 0.0, (), (), "low")

    kept_similarities = np.array([similarities[i] for i in keep], dtype=float)
    weights = kept_similarities / kept_similarities.sum()
    betas = np.array([catalogue_betas[i] for i in keep], dtype=float)

    blended = float(np.sum(weights * betas))
    spread = float(np.sqrt(np.sum(weights * (betas - blended) ** 2)))

    # Confidence tests the raw similarity of the closest neighbour, not its
    # normalised weight: weights sum to 1, so with four equally-good neighbours
    # each weight is 0.25 and a weight-based threshold would downgrade the very
    # case where the neighbourhood is strongest.
    confidence = (
        "high"
        if spread <= max_beta_sd
        and len(keep) >= min(3, k)
        and float(kept_similarities.max()) >= min_similarity
        else "low"
    )

    return ColdStartPrior(
        beta=blended,
        beta_sd=spread,
        neighbours=tuple(catalogue_skus[i] for i in keep),
        similarities=tuple(float(similarities[i]) for i in keep),
        confidence=confidence,
    )


class EmbeddingEncoder(Protocol):
    """Anything that turns product text into vectors."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]: ...


@dataclass
class TransformerTextEncoder:
    """Sentence-transformer encoder for product titles and categories.

    GPU-only, like every other model in PrismPrice. Encoding a catalogue on CPU
    is slow enough to turn a nightly refresh into a daily one, and the vectors
    feed a prior that sets prices, so they fall under the same reproducibility
    contract as everything else.

    The dependency is imported lazily: the cold-start *logic* in :func:`knn_prior`
    is useful without it, and most installs will not have it.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    _model: Any = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model

        device = require_gpu("features.embeddings")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "TransformerTextEncoder needs sentence-transformers. "
                'Install the modelling extra: pip install -e ".[modelling]"'
            ) from exc

        self._model = SentenceTransformer(self.model_name, device=str(device))
        return self._model

    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Encode and L2-normalise, so cosine similarity is a dot product."""
        model = self._load()
        vectors = model.encode(
            list(texts), batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return normalise(vectors)
