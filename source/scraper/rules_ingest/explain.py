"""Explain an upsert collision: why did two statements land on one subject?

Every collision in a run marks something that went wrong upstream -- the
extractor named a different fact with a name that belongs to another, misread
a number, invented a statement, or the merge judge folded two facts into one.
The run report shows both sides; this asks the 235B to say which of those it
was, which side is right and what the names or values should have been, and
the report prints the answer under the collision.

Probed on the 16 collisions of one five-page run (experiments.md 2026-09-04
§15): every judge-side collision was diagnosed correctly, extractor-side ones
were not -- the model did not know that an alias hit involves no judge, did
not know the naming shape, and did not know הונגשו means "made accessible".
So the prompt below tells it all three. Its answers are advisory: nothing
downstream acts on them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import OpenAI
from pydantic import BaseModel, field_validator

from db.models import QualifierUnit
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_MODEL,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)

if TYPE_CHECKING:
    from source.scraper.rules_ingest.db import DroppedRule, ResolvedRule
    from source.scraper.rules_ingest.ingest import SiteReport
    from source.scraper.subjects.resolve import ResolutionTrace

CAUSES = (
    "extractor_wrong_name",
    "extractor_wrong_value",
    "extractor_hallucination",
    "judge_over_merge",
    "true_duplicate",
    "other",
)

EXPLAIN_SYSTEM_PROMPT = """You review a campsite-rules ingestion pipeline and explain one collision.

How the pipeline works:
- An extractor reads a Hebrew section of a campsite page and emits statements: a
  snake_case English subject name, a polarity (true/false) or a number with a unit,
  and the sentence it read (verbatim). Subject names have ONE shape:
    <topic>[_<scope>]_<predicate>   for a rule, predicate one of allowed, required,
                                    time, fee_ils, fee_percent, min_age, max_age,
                                    min_nights, max_nights, min_occupancy,
                                    max_occupancy, count
    <thing>[_in_<place>]            for an amenity: a bare noun, no predicate; a count
                                    is the statement's NUMBER, never part of the name
                                    (never propose `toilets_count`).
  A property stated about a list goes into every name on it: "הונגשו X, Y, Z"
  (X, Y, Z were made accessible) yields accessible_x, accessible_y, accessible_z,
  never the bare nouns. A scope goes between topic and predicate:
  late_check_out_on_saturday_evening_allowed is a different subject from
  late_check_out_allowed. Text about one room or unit type is ignored: the page is
  read for the campsite as a whole.
- A resolver maps each subject name onto the vocabulary. First an exact alias
  lookup: if the name is already an alias of a subject it is taken WITHOUT any
  model call ("alias hit"). Otherwise the nearest existing subjects of the same
  category are shown to a merge judge model, which says whether the new name is
  the same subject as one of them ("ADJUDICATOR merged into") or not ("INSERTED").
- The database allows ONE row per subject per campsite. When two statements from
  one page land on the same subject, the second is refused: "duplicate" if it
  states the same thing, "CONFLICTING" if polarity or number differ.

So: an alias hit on a bare name that should have carried a qualifying word is an
EXTRACTOR error (the judge never ran); a merge of two names that differ by a
scope or qualifying word is a JUDGE error; a statement whose sentence does not
contain the thing named is a hallucination (חושה / חושות is a hut, an
accommodation unit -- not senses, not a fountain).

Output valid JSON only:
{
  "cause": "extractor_wrong_name" | "extractor_wrong_value" | "extractor_hallucination"
           | "judge_over_merge" | "true_duplicate" | "other",
  "which_is_right": "kept" | "dropped" | "both" | "neither",
  "explanation": one or two English sentences citing the Hebrew,
  "fix": the subject name(s) or value(s) that should have been produced, in the
         shape above
}
"""


class ConflictExplanationPayload(BaseModel):
    cause: str = "other"
    which_is_right: str = "neither"
    explanation: str = ""
    fix: str = ""

    @field_validator("cause", mode="before")
    @classmethod
    def _known_cause(cls, value: object) -> str:
        text = str(value or "").strip().casefold()
        return text if text in CAUSES else "other"

    @field_validator("which_is_right", mode="before")
    @classmethod
    def _known_side(cls, value: object) -> str:
        text = str(value or "").strip().casefold()
        return text if text in ("kept", "dropped", "both", "neither") else "neither"


@dataclass(frozen=True)
class ConflictExplanation:
    cause: str
    which_is_right: str
    explanation: str
    fix: str

    def one_line(self) -> str:
        return f"{self.cause}, right: {self.which_is_right} — {self.explanation} Fix: {self.fix}"


class ConflictExplainerLLMClient:
    """Asks the 235B why two statements collided. Advisory only."""

    MODEL = QWEN_INSTRUCT_MODEL
    TEMPERATURE = 0
    ROLE = "conflict_explainer"

    def __init__(self, client: OpenAI | None = None, *, model: str | None = None) -> None:
        self._client = client
        self.model = model or self.MODEL

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_nebius_openai_client()
        return self._client

    def explain(
        self,
        drop: DroppedRule,
        *,
        kept_how: str | None = None,
        dropped_how: str | None = None,
        usage: LlmUsage | None = None,
    ) -> ConflictExplanation:
        body = "\n".join(
            [
                f"Collision ({drop.label}) on campsite {drop.campsite_id}:",
                "",
                *_side("kept", drop.kept, kept_how),
                *_side("dropped", drop.dropped, dropped_how),
            ]
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": body},
            ],
            temperature=self.TEMPERATURE,
        )
        if usage is not None:
            usage.add_chat(response.usage, role=self.ROLE, model=self.model)
        try:
            data = _parse_json_payload(response.choices[0].message.content or "")
        except ValueError:
            data = {"explanation": (response.choices[0].message.content or "").strip()}
        payload = ConflictExplanationPayload.model_validate(data)
        return ConflictExplanation(
            payload.cause, payload.which_is_right, payload.explanation, payload.fix
        )


def _side(role: str, rule: ResolvedRule, how: str | None) -> list[str]:
    where = f" (section {rule.section_title!r})" if rule.section_title else ""
    return [
        f"{role.upper()}: subject name {rule.term!r}{where} -> {_value(rule)}",
        f"  sentence: {rule.evidence_span!r}",
        f"  resolver: {how or '(resolved from the page cache)'}",
        "",
    ]


def _value(rule: ResolvedRule) -> str:
    if rule.qualifier is None:
        return f"polarity={rule.polarity}"
    try:
        unit = QualifierUnit(int(rule.qualifier_unit)).name.lower()
    except ValueError:
        unit = str(rule.qualifier_unit)
    return f"polarity={rule.polarity} qualifier={rule.qualifier} {unit}"


def explain_conflicts(
    report: SiteReport,
    explainer: ConflictExplainerLLMClient,
    *,
    usage: LlmUsage | None = None,
    verbose: bool = True,
) -> int:
    """Fill `report.explanations` for every drop; returns how many were explained.

    One model call per collision, tagged `conflict_explainer` in the cost
    report. A failed call leaves that collision unexplained rather than
    stopping the page.
    """
    if not report.drops:
        return 0
    first: dict[str, ResolutionTrace] = {}
    for trace in report.traces:
        first.setdefault(trace.term, trace)
    if verbose:
        print(
            f"    explaining {len(report.drops)} collision(s) -> {explainer.model} ",
            end="",
            flush=True,
        )
    started = time.monotonic()
    done = 0
    for index, drop in enumerate(report.drops):
        kept = first.get(drop.kept.term or "")
        dropped = first.get(drop.dropped.term or "")
        try:
            report.explanations[index] = explainer.explain(
                drop,
                kept_how=kept.outcome if kept else None,
                dropped_how=dropped.outcome if dropped else None,
                usage=usage,
            )
            done += 1
            if verbose:
                print(".", end="", flush=True)
        except Exception as exc:  # noqa: BLE001 -- advisory; never stop the page
            if verbose:
                print(f"\n    explainer failed on collision {index}: {exc}", end="")
    if verbose:
        print(f" {done} explained in {time.monotonic() - started:.1f}s")
    return done
