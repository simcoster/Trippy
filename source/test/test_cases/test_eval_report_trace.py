"""Eval report traces extractor, planner queries, RAG, and judge."""

from pathlib import Path

from source.agent.claim_judge import apply_claim_rule_judgements
from source.eval.run import write_report


def test_judge_attaches_retrieved_claims_and_rules():
    fit = {
        "campsite_id": 14,
        "campsite": "Masada",
        "why": [{"query": "near the sea", "claim": "region:dead-sea"}],
        "review_claims": [
            {
                "query": "near the sea",
                "claim": "region:dead-sea",
                "is_positive": True,
            }
        ],
    }

    def _judge(*, query, campsite, claims, rules, usage=None):
        return {
            "relevant_claims": ["region:dead-sea"],
            "satisfies": True,
            "satisfy_by": "claim",
            "reason": "dead sea slug",
        }

    def _rules(query, **kwargs):
        return [
            {
                "campsite_id": 14,
                "subject": "tent_pitch",
                "polarity": True,
                "evidence_span": "לינת שטח",
            }
        ]

    out = apply_claim_rule_judgements(
        {"fits": [fit], "rejected": [], "rejected_count": 0},
        judge=_judge,
        search_rules=_rules,
    )
    retrieved = out["fits"][0]["retrieved"]
    assert retrieved[0]["query"] == "near the sea"
    assert retrieved[0]["claims"] == [
        {"claim": "region:dead-sea", "is_positive": True}
    ]
    assert retrieved[0]["rules"][0]["subject"] == "tent_pitch"
    assert retrieved[0]["rules"][0]["evidence_span"] == "לינת שטח"


def test_report_includes_extract_queries_rag_and_judge(tmp_path: Path):
    path = tmp_path / "report.md"
    rows = [
        {
            "id": "E03",
            "difficulty": "easy",
            "query": "קמפינג באוהל עד 80 שקל לאדם",
            "seconds": 1.2,
            "score": {
                "ok": False,
                "failures": ["party_size None != 1"],
                "extract_date": {"start": "2026-09-17", "end": "2026-09-18"},
                "fit_sites": [14],
            },
            "extract": {
                "date": {"start": "2026-09-17", "end": "2026-09-18"},
                "numeric_constraints": [
                    {"field": "price_per_night", "operator": "<=", "value": 80}
                ],
                "semantic_constraints": [
                    {"query": "camping", "locus": "site"},
                    {"query": "tent", "locus": "site"},
                ],
            },
            "planner": {
                "fits": [
                    {
                        "campsite_id": 14,
                        "campsite": "Masada",
                        "accommodation_type": "לינת שטח באוהלים פרטיים",
                        "price_per_night": 64.0,
                        "why": [
                            {
                                "query": "tent",
                                "stated_amenity": "tent_pitch",
                            }
                        ],
                        "retrieved": [
                            {
                                "query": "tent",
                                "claims": [
                                    {
                                        "claim": "Great tent camping",
                                        "is_positive": True,
                                    }
                                ],
                                "rules": [
                                    {
                                        "subject": "tent_pitch",
                                        "polarity": True,
                                        "evidence_span": "לינת שטח",
                                    }
                                ],
                            }
                        ],
                        "claim_judge": [
                            {
                                "query": "tent",
                                "satisfies": True,
                                "satisfy_by": "rule",
                                "relevant_claims": ["Great tent camping"],
                                "reason": "official tent pitch",
                            }
                        ],
                    }
                ],
                "rejected": [],
            },
        }
    ]
    write_report(path, {"id": "planner_v1", "as_of": "2026-09-08"}, rows, 1.2)
    text = path.read_text(encoding="utf-8")
    assert "party_size None != 1" in text
    assert "numeric: price_per_night <= 80" in text
    assert "semantic: camping (site); tent (site)" in text
    assert "planner queries: camping, tent" in text
    assert "RAG `tent` claims: Great tent camping pos=True" in text
    assert "rules: tent_pitch pol=True “לינת שטח”" in text
    assert "judge `tent` satisfies=True by=rule relevant: Great tent camping" in text
