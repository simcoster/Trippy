"""Reading one page on behalf of one subcamp.

Akhziv is a single page describing two separately-run subcamps, `חניון צפוני`
and `חניון דרומי`, each with a full amenity list of its own. Every count is
different, and `campsite_rules` allows one row per (campsite, subject), so
ingesting the page once drops the second list as CONFLICTING.

The fix is to ingest it once *per subcamp*, into that subcamp's own `campsites`
row. This module is the part that decides what each pass reads and how it is
told which subcamp it is reading for. Nothing here names a subcamp: a `Subcamp`
is built from the `campsites.subcamp` JSONB, seeded from `config.json`.

Two things do the work, and the division between them was measured rather than
guessed (docs/design.md):

`hybrid_slice` keeps **both** amenity lists in the text and cuts only the lines
the prompt provably cannot handle. Cutting the section down to one subcamp's
lines separated them perfectly and then destroyed the gendered counts — shown
only its own list the extractor stored `shower_stalls_women = 11` (7 + 4) and
`toilet_stalls_men = 18` (10 + 8) in four interleaved rounds out of four, where
showing it both lists got them right 4/4. With both present it has to tell
`מקלחות (7 … 4 …)` from `מקלחות מים חמים (9 … 9 …)`, and per-gender subjects are
the only way; alone, it totals them.

`subcamp_prompt` then does the separating, which it is good at: over the whole
page it took nothing from the other subcamp's list.

Together they scored 33 of the 34 subcamp-owned lines exactly right against the
page's own markup, with no leaks and no wrong numbers — the best of four
strategies tried.
"""

from __future__ import annotations

from dataclasses import dataclass

from source.scraper.rules_ingest.llm import SYSTEM_PROMPT
from source.scraper.rules_ingest.sections import Section

# A bare heading is at most this many words. `חניון צפוני` is two; a line like
# `גז לקבוצות (חניון ראשי 2 מבערים + חניון משני 2 מבערים)` names subcamps but is
# a statement, not a heading, and must not switch anything.
MAX_HEADING_WORDS = 3


@dataclass(frozen=True)
class Subcamp:
    """One subcamp of a split site: its campsites row and the words the page uses."""

    campsite_id: int
    name: str
    heading: str
    aliases: tuple[str, ...] = ()
    unit_name_contains: tuple[str, ...] = ()
    default_units: bool = False

    @property
    def words(self) -> tuple[str, ...]:
        """Every way the page names this subcamp, longest first.

        Longest first because the forms nest — `חניון הצפוני` contains
        `חניון צפוני` only loosely, but a clause cut must strip the longest
        label it actually starts with.
        """
        return tuple(sorted({self.heading, *self.aliases}, key=len, reverse=True))

    @classmethod
    def from_row(cls, campsite_id: int, name: str, subcamp: dict) -> Subcamp:
        return cls(
            campsite_id=campsite_id,
            name=name,
            heading=subcamp["heading"],
            aliases=tuple(subcamp.get("aliases") or ()),
            unit_name_contains=tuple(subcamp.get("unit_name_contains") or ()),
            default_units=bool(subcamp.get("default_units")),
        )


def load_subcamps(conn, campsite_id: int) -> list[Subcamp]:
    """A site's subcamps, straight from its child rows. Empty for a normal site."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, subcamp FROM campsites "
            "WHERE parent_id = %s ORDER BY id",
            (campsite_id,),
        )
        return [Subcamp.from_row(*row) for row in cur.fetchall()]


def subcamps_named_in(text: str, subcamps: list[Subcamp]) -> set[int]:
    """Which subcamps a piece of text names outright, by campsite id."""
    return {s.campsite_id for s in subcamps if any(w in text for w in s.words)}


def subcamp_prompt(subcamp: Subcamp, subcamps: list[Subcamp]) -> str:
    """The production extractor prompt with a subcamp filter in front of it.

    In front, not appended: the production prompt ends with its output schema,
    and an instruction placed after that reads as a note on the schema rather
    than as the frame for the whole task.
    """
    others = [s for s in subcamps if s.campsite_id != subcamp.campsite_id]
    other_names = ", ".join(f"`{s.heading}`" for s in others)
    other_words = ", ".join(f"`{w}`" for s in others for w in s.words)
    return f"""SUBCAMP FILTER — apply this before every other rule below.

This page describes SEPARATELY-RUN areas of one site. You are extracting for
`{subcamp.heading}` ONLY. The other area(s) on this page: {other_names}.

- A heading naming another area starts that area's list. Every line under it,
  until the next heading, is NOT yours. Skip all of it, however similar it looks
  to a line you already read.
- A heading `{subcamp.heading}` starts YOUR list. Extract every line under it.
- A statement naming another area anywhere in it — {other_words} — is NOT yours.
- A statement naming YOUR area is yours. Drop the area word from the subject
  name: it is understood. Never emit a subject naming an area.
- A statement naming NO area applies to the whole site and IS yours. Rental
  equipment, opening hours, arrival and departure times, dog policy and rate
  notes are of this kind — extract them normally.
- An area heading is a heading, not an amenity. Never emit a statement for it.
- Another area's numbers must never appear in your output. If you are unsure
  which area a line belongs to, prefer to emit it: a fact stated once for the
  whole site is worth more than a fact dropped.

{SYSTEM_PROMPT}"""


def subcamp_clause(line: str, subcamp: Subcamp, subcamps: list[Subcamp]) -> str:
    """One subcamp's clause out of a line that states several.

    `נגישות` is a single running line — `אכזיב צפון: … אכזיב דרום: …` — with no
    markup to cut on, and the prompt does not hold the halves apart: both passes
    quoted the whole line, and both emitted the northern
    `שביל לאזור הקמת האוהלים`.
    """
    marks = sorted(
        (index, s.campsite_id)
        for s in subcamps
        for word in s.words
        if (index := line.find(word)) >= 0
    )
    for position, (index, campsite_id) in enumerate(marks):
        if campsite_id != subcamp.campsite_id:
            continue
        end = marks[position + 1][0] if position + 1 < len(marks) else len(line)
        clause = line[index:end].strip(" ,.:")
        # Drop the label that opened the clause. Left in, the extractor reads
        # `אכזיב צפון:` as a fact and emits a subject for the area itself —
        # `northern_parking_area` and `south_parking_lot` both reached the
        # production dictionary that way, from the unsplit run.
        for word in subcamp.words:
            if clause.startswith(word):
                return clause[len(word) :].strip(" ,.:")
        return clause
    return ""


def hybrid_slice(text: str, subcamp: Subcamp, subcamps: list[Subcamp]) -> str:
    """The text this subcamp's pass should read.

    Both per-subcamp lists are kept whole — see the module docstring for why
    cutting them costs the gendered counts. Only two shapes are cut: a line that
    mentions another subcamp while saying something else
    (`מרכז שירות למבקר (בחניון הצפוני)`, which the southern pass claimed
    otherwise), and a line naming several subcamps at once.
    """
    kept: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        named = subcamps_named_in(stripped, subcamps)
        if len(named) > 1:
            clause = subcamp_clause(stripped, subcamp, subcamps)
            if clause:
                kept.append(clause)
            continue
        # A bare heading is the section's structure and stays for every pass, so
        # the extractor can still see which list is which.
        is_heading = len(named) == 1 and len(stripped.split()) <= MAX_HEADING_WORDS
        if named and subcamp.campsite_id not in named and not is_heading:
            continue
        kept.append(stripped)
    return "\n".join(kept)


def subcamp_sections(
    sections: list[Section], subcamp: Subcamp, subcamps: list[Subcamp]
) -> list[Section]:
    """Every section cut down for one subcamp; sections left empty are dropped."""
    out = []
    for section in sections:
        text = hybrid_slice(section.text, subcamp, subcamps)
        if text:
            out.append(Section(section.title, text, section.source_url))
    return out


def unit_owner(unit_name: str, campsite_id: int, subcamps: list[Subcamp]) -> int:
    """Which campsite row owns a booking unit of this name.

    The booking engine returns one flat list of unit types for the whole site —
    it has one hotel id, and knows nothing about subcamps. Akhziv's two tent
    pitches name theirs (`לינת שטח באוהלים פרטיים - חניון צפוני`); its four
    `חושה` types name none at all, so `default_units` places them. The northern
    subcamp has only tents, which is why "everything unnamed is southern" is a
    complete rule here and not a guess.

    FIXME(subcamp-routing): substring-matching a booking unit name is the weak
    link in the whole subcamp design. It works because Akhziv's operator happens
    to put the subcamp in the unit name, and it will not survive them renaming a
    unit, nor a second split site whose units are distinguished some other way —
    and it fails silently, by quietly routing everything to the default. Worth
    replacing with something that reconciles booking units against the info
    page's per-subcamp lodging panel (which does state the split) rather than
    against a configured string. Config keeps it honest meanwhile: the strings
    live in `campsites.subcamp`, seeded from `config.json`, not in code.
    """
    if not subcamps:
        return campsite_id
    for subcamp in subcamps:
        if any(needle in unit_name for needle in subcamp.unit_name_contains):
            return subcamp.campsite_id
    for subcamp in subcamps:
        if subcamp.default_units:
            return subcamp.campsite_id
    # No default configured: the parent keeps the unit rather than dropping it.
    return campsite_id


def owned_site_ids(campsite_id: int, subcamps: list[Subcamp]) -> list[int]:
    """The parent and every subcamp, for queries that must span a split site."""
    return [campsite_id, *(s.campsite_id for s in subcamps)]


def group_by_owner(
    unit_names, campsite_id: int, subcamps: list[Subcamp]
) -> dict[int, list[str]]:
    """Unit names bucketed by owning campsite, parent-first, order preserved."""
    grouped: dict[int, list[str]] = {}
    for name in unit_names:
        grouped.setdefault(unit_owner(name, campsite_id, subcamps), []).append(name)
    return grouped
