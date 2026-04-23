from io import StringIO

from bbv.federated.progress import progress_iterable


def test_progress_iterable_writes_description_when_enabled() -> None:
    stream = StringIO()

    items = list(progress_iterable(range(3), description="Rounds", enabled=True, stream=stream))

    assert items == [0, 1, 2]
    assert "Rounds" in stream.getvalue()


def test_progress_iterable_is_silent_when_disabled() -> None:
    stream = StringIO()

    items = list(progress_iterable(range(2), description="Clients", enabled=False, stream=stream))

    assert items == [0, 1]
    assert stream.getvalue() == ""
