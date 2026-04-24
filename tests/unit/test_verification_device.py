from __future__ import annotations

import pytest

from bbv.verification.baseline import _resolve_verification_device


def test_resolve_verification_device_auto_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("bbv.verification.baseline.torch.cuda.is_available", lambda: True)

    resolved = _resolve_verification_device("auto")

    assert resolved.type == "cuda"


def test_resolve_verification_device_cuda_requires_available_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("bbv.verification.baseline.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested"):
        _resolve_verification_device("cuda")
