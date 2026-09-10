"""Eval prints and reports token counts from extractor + judge."""

from pathlib import Path
from types import SimpleNamespace

from source.eval.run import format_usage_line, write_report
from source.scraper.amenity_enrichment.llm import langchain_chat_usage


def test_format_usage_line_includes_roles():
    line = format_usage_line(
        {
            "input_tokens": 1000,
            "output_tokens": 80,
            "by_role": [
                {
                    "role": "extract",
                    "input_tokens": 200,
                    "output_tokens": 20,
                    "calls": 1,
                },
                {
                    "role": "claim_judge",
                    "input_tokens": 800,
                    "output_tokens": 60,
                    "calls": 10,
                },
            ],
        }
    )
    assert line == (
        "tokens in=1000 out=80 extract in=200 out=20 "
        "claim_judge in=800 out=60×10"
    )


def test_langchain_usage_from_usage_metadata():
    raw = langchain_chat_usage(
        SimpleNamespace(usage_metadata={"input_tokens": 12, "output_tokens": 3})
    )
    assert raw.prompt_tokens == 12
    assert raw.completion_tokens == 3


def test_report_includes_token_counts(tmp_path: Path):
    path = tmp_path / "report.md"
    rows = [
        {
            "id": "E02",
            "difficulty": "easy",
            "query": "קמפינג לזוג",
            "seconds": 10.0,
            "score": {
                "ok": True,
                "failures": [],
                "extract_date": {"start": "2026-09-10", "end": "2026-09-11"},
                "fit_sites": [1],
            },
            "extract": {},
            "planner": {"fits": [], "rejected": []},
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 80,
                "by_role": [
                    {
                        "role": "extract",
                        "input_tokens": 200,
                        "output_tokens": 20,
                        "calls": 1,
                    },
                    {
                        "role": "claim_judge",
                        "input_tokens": 800,
                        "output_tokens": 60,
                        "calls": 10,
                    },
                ],
            },
        }
    ]
    write_report(path, {"id": "planner_v1", "as_of": "2026-09-08"}, rows, 10.0)
    text = path.read_text(encoding="utf-8")
    assert "tokens in=1000 out=80" in text
    assert "## Tokens" in text
    assert "| E02 | 1000 | 80 | 200/20 | 800/60×10 |" in text
    assert "| **total** | 1000 | 80 | 200/20 | 800/60×10 |" in text
