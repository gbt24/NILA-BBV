from __future__ import annotations

import pytest

from bbv.attacks.suite import _resolve_attack_device


def test_resolve_attack_device_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("bbv.attacks.suite.torch.cuda.is_available", lambda: False)

    resolved = _resolve_attack_device("auto")

    assert resolved.type == "cpu"


def test_resolve_attack_device_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        _resolve_attack_device("tpu")
