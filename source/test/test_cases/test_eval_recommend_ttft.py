"""Eval dumps recommend TTFT next to the recommend wall."""

from pathlib import Path

from source.agent.recommender import Recommendation, RecommendResult
from source.eval.run import (
    format_recommend_ttft,
    planner_pack,
    replay_recommend_rows,
    write_report,
)


def _fit() -> dict:
    return {
        "campsite_id": 37,
        "campsite": "אכזיב צפון",
        "accommodation_type": "אוהל",
        "start": "2026-09-10",
        "end": "2026-09-11",
        "price_per_night": 150,
    }


def _result(**overrides) -> RecommendResult:
    rec = Recommendation(
        campsite_id=37,
        campsite="אכזיב צפון",
        accommodation_type="אוהל",
        start="2026-09-10",
        end="2026-09-11",
        price_per_night=150,
        why="ים",
    )
    row = {
        "recommendations": (rec,),
        "empty": None,
        "text": "ים",
        "ttft_chunk_ms": 2100.4,
        "ttft_spoken_ms": 8400.1,
        "elapsed_ms": 43900.0,
    }
    row.update(overrides)
    return RecommendResult(**row)


def test_replay_dumps_recommend_ttft():
    pack = planner_pack("ליד הים", {}, {"fits": [_fit()]})

    def _fake(query: str, payload: dict) -> RecommendResult:
        return _result()

    src = {
        "id": "E04",
        "difficulty": "easy",
        "query": "ליד הים",
        "score": {"ok": True, "failures": [], "fit_sites": [37]},
        "pack": pack,
    }
    out = replay_recommend_rows([src], recommend_fn=_fake)
    rec = out[0]["recommend"]
    assert rec["ttft_chunk_ms"] == 2100.4
    assert rec["ttft_spoken_ms"] == 8400.1
    assert "elapsed_ms" not in rec


def test_format_recommend_ttft():
    assert (
        format_recommend_ttft(
            {"ttft_chunk_ms": 2100, "ttft_spoken_ms": 8400}
        )
        == "ttft_chunk=2.1s ttft_spoken=8.4s"
    )
    assert format_recommend_ttft({}) == ""


def test_write_report_timing_includes_ttft(tmp_path: Path):
    path = tmp_path / "report.md"
    rows = [
        {
            "id": "E04",
            "difficulty": "easy",
            "query": "ליד הים",
            "seconds": 43.9,
            "score": {"ok": True, "failures": [], "fit_sites": [37]},
            "extract": {},
            "planner": {"fits": [], "rejected": []},
            "stages": {"recommend": {"s": 43.9, "n": 1}},
            "recommend": {
                "recommendations": [],
                "text": "ים",
                "ttft_chunk_ms": 2100,
                "ttft_spoken_ms": 8400,
            },
        }
    ]
    write_report(path, {"id": "planner_v1", "as_of": "2026-09-08"}, rows, 43.9)
    text = path.read_text(encoding="utf-8")
    assert "ttft_chunk=2.1s ttft_spoken=8.4s" in text
    assert (
        "| id | s | extract | sql | embed | retrieve | rules | judge | "
        "recommend | ttft_chunk | ttft_spoken |"
    ) in text
    assert "| E04 | 43.9 |  |  |  |  |  |  | 43.9 | 2.1 | 8.4 |" in text
