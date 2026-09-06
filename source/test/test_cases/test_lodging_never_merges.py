"""One `<h4>` is one accommodation type. Always.

The segmenter normalises unit names, which is what turns
`חדר צוות מאובזר ומונגש חדר מספר 5` into a product name instead of an instance
identifier. Metsada is where that becomes dangerous: two of its staff rooms
differ only by their room number, and they sleep 7 and 5. Normalise both and
they collide on `(hotel_id, name)`, and one silently overwrites the other.

So the never-merge rule is enforced in code, over whatever the model returns —
`assemble_units` is what these cover. No LLM and no database: the segmenter's
answer is supplied directly, including answers a model should never give.
"""

from __future__ import annotations

from pathlib import Path

from source.scraper.rules_ingest.lodging import (
    assemble_units,
    parse_lodging_blocks,
)

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "info_site"
METSADA = (_FIXTURES / "metsada_lodging_panel.html").read_text(encoding="utf-8")

ROOM_6 = "חדר צוות מאובזר ומונגש חדר מספר 6"
ROOM_1 = "חדר צוות מאובזר ומונגש חדר מספר 1"
NORMALISED = "חדר צוות מאובזר ומונגש"


def metsada_blocks():
    return parse_lodging_blocks(METSADA)


def answer_naming(blocks, names: dict[str, str]) -> dict:
    """A segmenter answer that renames headings and claims the paragraph after each."""
    units = []
    for block in blocks:
        if block.kind != "heading":
            continue
        following = [
            b.index
            for b in blocks
            if b.kind == "para" and b.index == block.index + 1
        ]
        units.append(
            {
                "heading": block.index,
                "name": names.get(block.text, block.text),
                "blocks": following,
            }
        )
    return {"units": units, "rules": [], "notes": []}


def test_the_two_metsada_rooms_stay_two_units_when_normalised_alike():
    """The failure this file exists for: the model normalises both room names to
    the same string. Both must survive, and with different names."""
    blocks = metsada_blocks()
    seg = assemble_units(
        blocks, answer_naming(blocks, {ROOM_6: NORMALISED, ROOM_1: NORMALISED})
    )
    accessible = [u for u in seg.units if "ומונגש" in u.name]
    assert len(accessible) == 2, [u.name for u in seg.units]
    assert len({u.name for u in accessible}) == 2, [u.name for u in accessible]


def test_the_colliding_pair_keeps_its_identifiers():
    blocks = metsada_blocks()
    seg = assemble_units(
        blocks, answer_naming(blocks, {ROOM_6: NORMALISED, ROOM_1: NORMALISED})
    )
    names = {u.name for u in seg.units}
    assert ROOM_6 in names and ROOM_1 in names


def test_a_model_that_merges_outright_still_yields_both_units():
    """Not a rename but an omission: the answer mentions only one of the two."""
    blocks = metsada_blocks()
    answer = answer_naming(blocks, {})
    answer["units"] = [u for u in answer["units"] if ROOM_1 not in u["name"]]
    seg = assemble_units(blocks, answer)
    headings = [b for b in blocks if b.kind == "heading"]
    assert len(seg.units) == len(headings)
    assert ROOM_1 in {u.name for u in seg.units}


def test_the_two_rooms_keep_the_capacities_the_panel_states():
    """Room 6 sleeps 7 and room 1 sleeps 5 — the reason merging them loses data."""
    blocks = metsada_blocks()
    seg = assemble_units(blocks, answer_naming(blocks, {}))
    by_name = {u.name: u for u in seg.units}
    assert "עד 7 לנים" in by_name[ROOM_6].text
    assert "עד 5 לנים" in by_name[ROOM_1].text


def test_a_name_that_does_not_collide_keeps_its_normalisation():
    """Only a collision forces the identifier back; everything else stays clean."""
    blocks = metsada_blocks()
    seg = assemble_units(
        blocks, answer_naming(blocks, {ROOM_6: NORMALISED, ROOM_1: "חדר צוות אחר"})
    )
    names = {u.name for u in seg.units}
    assert NORMALISED in names
    assert "חדר צוות אחר" in names


def test_a_paragraph_claimed_by_a_unit_is_not_also_a_rule():
    blocks = metsada_blocks()
    answer = answer_naming(blocks, {})
    claimed = answer["units"][0]["blocks"]
    answer["rules"] = list(claimed)
    seg = assemble_units(blocks, answer)
    assert seg.rules == []
