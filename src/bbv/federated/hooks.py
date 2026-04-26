from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import torch

from bbv.watermarking.codebook import (
    generate_codebook,
    generate_hadamard_codebook,
    generate_single_trigger_codebook,
)
from bbv.watermarking.queries import adapt_queries_to_shape, build_negative_queries, build_positive_queries


@dataclass(frozen=True)
class WatermarkHook:
    owner_id: str
    wm_weight: float
    codebook: list[int] = field(init=False)
    positive_queries: list[torch.Tensor] = field(init=False)
    negative_queries: list[torch.Tensor] = field(init=False)
    code_length: InitVar[int]
    seed: InitVar[int]
    codebook_type: InitVar[str] = "multi-bit"

    def __post_init__(
        self, code_length: int, seed: int, codebook_type: str = "multi-bit"
    ) -> None:
        if codebook_type == "single-trigger":
            codebook = generate_single_trigger_codebook(code_length)
        elif codebook_type == "hadamard":
            owner_index = sum(ord(ch) for ch in self.owner_id) % code_length
            codebook = generate_hadamard_codebook(owner_index, code_length)
        else:
            codebook = generate_codebook(
                owner_id=self.owner_id,
                code_length=code_length,
                seed=seed,
            )
        object.__setattr__(self, "codebook", codebook)
        object.__setattr__(
            self,
            "positive_queries",
            build_positive_queries(codebook=codebook, seed=seed),
        )
        object.__setattr__(
            self,
            "negative_queries",
            build_negative_queries(codebook=codebook, seed=seed),
        )

    def with_weight(self, wm_weight: float) -> "WatermarkHook":
        updated = WatermarkHook(
            owner_id=self.owner_id,
            code_length=len(self.codebook),
            wm_weight=wm_weight,
            seed=0,
        )
        object.__setattr__(updated, "codebook", self.codebook)
        object.__setattr__(updated, "positive_queries", self.positive_queries)
        object.__setattr__(updated, "negative_queries", self.negative_queries)
        return updated

    def build_query_batch(self, feature_shape: torch.Size) -> torch.Tensor:
        return torch.stack(adapt_queries_to_shape(self.positive_queries, feature_shape))
