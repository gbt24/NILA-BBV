from __future__ import annotations

import pytest

from bbv.federated.fedavg import _resolve_training_device


def test_resolve_training_device_auto_prefers_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bbv.federated.fedavg.torch.cuda.is_available", lambda: True)

    resolved = _resolve_training_device("auto")

    assert resolved.type == "cuda"


def test_resolve_training_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bbv.federated.fedavg.torch.cuda.is_available", lambda: False)

    resolved = _resolve_training_device("auto")

    assert resolved.type == "cpu"


def test_resolve_training_device_raises_for_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("bbv.federated.fedavg.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested"):
        _resolve_training_device("cuda")
