"""Stage clock is a no-op unless collect_stages is active."""

from source.agent.timing import (
    STAGE_ORDER,
    collect_stages,
    format_stages,
    merge_snapshots,
    stage,
)


def test_stage_is_noop_without_collector():
    with stage("extract"):
        pass


def test_snapshot_lists_every_stage():
    with collect_stages() as clock:
        pass
    assert list(clock.snapshot()) == list(STAGE_ORDER)
    assert all(item["n"] == 0 for item in clock.snapshot().values())


def test_collect_sums_calls():
    with collect_stages() as clock:
        with stage("judge"):
            pass
        with stage("judge"):
            pass
        with stage("embed"):
            pass
    snap = clock.snapshot()
    assert snap["judge"]["n"] == 2
    assert snap["embed"]["n"] == 1
    assert snap["extract"]["n"] == 0
    text = format_stages(snap)
    assert "judge=" in text
    assert "×2" in text


def test_merge_snapshots_adds_seconds_and_calls():
    merged = merge_snapshots(
        [
            {"extract": {"s": 1.0, "n": 1}, "judge": {"s": 2.0, "n": 3}},
            {"extract": {"s": 0.5, "n": 1}, "judge": {"s": 4.0, "n": 2}},
        ]
    )
    assert merged["extract"] == {"s": 1.5, "n": 2}
    assert merged["judge"] == {"s": 6.0, "n": 5}
    assert merged["sql"] == {"s": 0.0, "n": 0}
