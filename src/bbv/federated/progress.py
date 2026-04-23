from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar
import sys

from tqdm import tqdm


T = TypeVar("T")


def progress_iterable(
    iterable: Iterable[T],
    *,
    description: str,
    enabled: bool,
    stream: object | None = None,
    leave: bool = True,
    position: int = 0,
) -> Iterator[T]:
    if not enabled:
        return iter(iterable)
    return iter(
        tqdm(
            iterable,
            desc=description,
            file=stream if stream is not None else sys.stderr,
            leave=leave,
            position=position,
        )
    )
