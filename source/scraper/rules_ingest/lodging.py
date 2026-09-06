"""The `אפשרויות לינה` panel: a site's accommodation types, from the info page.

The panel is the operator's own statement of what the site has. The booking
engine only says what is free on the nights being scanned, so a fully-booked or
seasonally-closed unit is simply absent there, and it carries no inventory
counts — which is why types are read from here and availability is matched onto
them afterwards.

It is not server-rendered. `sections.parse_sections` drops it from the static
page on purpose; this module fetches it the way the page's own JavaScript does:

    GET /ajax-handler-wp-loadmore.php
        ?action=my_repeater_show_more
        &post_id=<body[data-id]>
        &offset=<the panel's own data-cnt>
        &nonce=<my_repeater_field_nonce, from an inline script>
    -> {"content": "<html>"}

The markup inside separates units from rules structurally, so no model is asked
which is which:

    <h4>בונגלו עם מזגן <b>(48)</b></h4>   a unit, with how many the site has
    <p>בכל בונגלו: 4 מיטות…</p>            that unit's description
    <p><strong>מותנה במינימום 2 לילות…</strong></p>   no <h4> owns it: a rule
    <h4> </h4>                            an empty separator

The rule paragraph sits *after* the bungalow's own paragraph, so "a paragraph
belongs to the heading above it" would file a site-wide minimum-nights rule
against bungalows. A heading claims its FIRST paragraph; anything after that
belongs to no unit.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from openai import OpenAI

from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    LlmUsage,
    _parse_json_payload,
    make_nebius_openai_client,
)
from source.scraper.info_site.parse import parse_wp_post_id
from source.scraper.rules_ingest.fetch import fetch_page_html

AJAX_URL = "https://www.parks.org.il/ajax-handler-wp-loadmore.php"
PANEL_TITLE = "אפשרויות לינה"

_NONCE_RE = re.compile(r'my_repeater_field_nonce["\']?\s*[:=]\s*["\']([a-z0-9]+)')
_COUNT_RE = re.compile(r"\((\d+)\)")
_WS_RE = re.compile(r"\s+")
_BLANK_LINES_RE = re.compile(r"\n{2,}")


def fold(text: str) -> str:
    """Panel/anchor text with format characters and NBSP out of the way.

    Some pages write the panel title with zero-width joiners inside it
    (`temp/pages/11.html`), so a plain `PANEL_TITLE in text` misses that site
    entirely and it silently yields no panel. Unicode category Cf covers the
    zero-width family; `normalize_label` handles only NBSP.
    """
    stripped = "".join(c for c in (text or "") if unicodedata.category(c) != "Cf")
    return _WS_RE.sub(" ", stripped.replace("\xa0", " ")).strip()


@dataclass(frozen=True)
class PanelRequest:
    """What the loadmore endpoint needs, all three read off the info page."""

    post_id: str
    offset: str
    nonce: str


@dataclass(frozen=True)
class LodgingUnit:
    """One accommodation type as the panel states it.

    `unit_count` is how many of this unit the site has — the `(48)` in the
    heading — and is None when the panel does not say. It is not `room_count`,
    which counts rooms inside one listing.

    `scope` is the `<h3>` in force above it. On a split site that is the subcamp
    the unit belongs to (`חניון צפוני` / `חניון דרומי` at Akhziv); elsewhere it
    is None. The panel stating the split in its own right is what
    `docs/design.md` calls the replacement worth building for
    `unit_name_contains` substring routing.
    """

    name: str
    unit_count: int | None
    text: str
    scope: str | None = None


@dataclass(frozen=True)
class Block:
    """One panel element, kept by index so nothing downstream rewrites its text.

    `kind` is "scope" for an `<h3>` (a subcamp heading), "heading" for an
    `<h4>` (a unit), "para" for a `<p>`.
    """

    index: int
    kind: str
    text: str
    unit_count: int | None = None
    scope: str | None = None


def parse_lodging_blocks(panel_html: str) -> list[Block]:
    """The panel as an ordered, indexed list of blocks. Pure BS4."""
    soup = BeautifulSoup(panel_html, "html.parser")
    root = soup.select_one("div.infoContent") or soup
    blocks: list[Block] = []
    scope: str | None = None
    for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "p"], recursive=False):
        if node.name == "p":
            text = _block_text(node)
            if text:
                blocks.append(Block(len(blocks), "para", text, None, scope))
            continue
        name, count = _heading(node)
        if not name or name == PANEL_TITLE:
            # An empty `<h4> </h4>` separator, or the panel's own title.
            continue
        if node.name != "h4":
            scope = name
            blocks.append(Block(len(blocks), "scope", name, None, scope))
            continue
        blocks.append(Block(len(blocks), "heading", name, count, scope))
    return blocks


def panel_request(html: str) -> PanelRequest | None:
    """The lodging panel's request triple, or None if the page does not offer it."""
    post_id = parse_wp_post_id(html) or ""
    nonce_match = _NONCE_RE.search(html)
    if not (post_id and nonce_match):
        return None
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.select("a[data-cnt]"):
        if PANEL_TITLE in fold(link.get_text(" ", strip=True)):
            offset = (link.get("data-cnt") or "").strip()
            # Read from the anchor, never assumed: panel order differs per site.
            if offset:
                return PanelRequest(post_id, offset, nonce_match.group(1))
    return None


def fetch_panel(site_url: str, html: str) -> str | None:
    """Fetch the panel body for an already-fetched info page."""
    request = panel_request(html)
    if request is None:
        return None
    query = urlencode(
        {
            "action": "my_repeater_show_more",
            "post_id": request.post_id,
            "offset": request.offset,
            "nonce": request.nonce,
        }
    )
    raw = fetch_page_html(f"{AJAX_URL}?{query}", referer=site_url)
    try:
        return json.loads(raw).get("content") or ""
    except json.JSONDecodeError:
        # Some responses come back as bare HTML rather than JSON.
        return raw


def _block_text(node) -> str:
    """Readable text, one line per `<br/>`-separated line, blank lines collapsed."""
    text = node.get_text("\n", strip=True)
    return _BLANK_LINES_RE.sub("\n", text).strip()


def _heading(node) -> tuple[str, int | None]:
    """A heading's unit name and its `(N)` count, the count taken out of the name.

    `<b>` is editorial bolding, not markup that means anything: across the 18
    sites it wraps the whole heading (`<b>עמדת חניה לקרוואן פרטי (9)</b>`), or
    only the count (`חדר צוות מאובזר<b> (5)</b>`), or the count plus the room
    numbers (`חדרי צוות <b>(2) חדרים 3-4</b>`). Reading the count out of the
    `<b>` and dropping the rest lost the unit name on seven sites and collapsed
    three distinct staff rooms onto one name at Metsada, so the tag is ignored
    and the whole heading text is read instead.

    The count is the first parenthesised number anywhere in the heading, not
    only a trailing one: `חדר צוות מאובזר ומונגש (1) חדר מספר 6` states both a
    count and a room number. A parenthetical without digits is part of the name
    (`מאהל גדול קבוע (מבנה כנעני)`).
    """
    text = fold(node.get_text(" ", strip=True))
    match = _COUNT_RE.search(text)
    if match is None:
        return text, None
    return fold(f"{text[: match.start()]} {text[match.end() :]}"), int(match.group(1))


def parse_lodging_panel(panel_html: str) -> tuple[list[LodgingUnit], list[str]]:
    """The panel's accommodation types, and the rule paragraphs that own no unit.

    Pure BS4 — no HTTP, no LLM, no DB — so it can be tested against a saved
    panel.
    """
    soup = BeautifulSoup(panel_html, "html.parser")
    root = soup.select_one("div.infoContent") or soup
    units: list[LodgingUnit] = []
    rules: list[str] = []
    # Only `<h4>` names a unit. `<h3>` is the panel's own title or, on a split
    # site, a subcamp heading -- Akhziv's `חניון צפוני` / `חניון דרומי` are
    # `<h3>`, and reading those as units invented two textless types.
    scope: str | None = None
    # The heading whose paragraph has not been taken yet. Cleared the moment it
    # claims one, which is what keeps a trailing paragraph from being filed
    # against the unit above it.
    pending: tuple[str, int | None] | None = None

    for node in root.find_all(["h1", "h2", "h3", "h4", "h5", "p"], recursive=False):
        if node.name == "p":
            text = _block_text(node)
            if not text:
                continue
            if pending is not None:
                name, count = pending
                units.append(LodgingUnit(name, count, text, scope))
                pending = None
            else:
                rules.append(text)
            continue
        name, count = _heading(node)
        if pending is not None:
            # A heading whose paragraph never arrived: Yehudia's whole panel is
            # one `<h4>` and nothing else. The unit is real; it has no text.
            units.append(LodgingUnit(pending[0], pending[1], "", scope))
            pending = None
        if node.name != "h4":
            scope = name if name and name != PANEL_TITLE else None
            continue
        # An empty `<h4> </h4>` is a separator; it opens nothing.
        pending = (name, count) if name else None

    if pending is not None:
        units.append(LodgingUnit(pending[0], pending[1], "", scope))
    return units, rules


SEGMENT_PROMPT = """You are labelling the blocks of a campsite's lodging panel.

You are given the panel as a numbered list. `H:` is a heading, `P:` is a
paragraph, `S:` is a sub-area heading. Every block keeps its number.

Decide, for each heading, which paragraphs describe THAT accommodation unit, and
which paragraphs are rules about the whole lodging area rather than one unit.

Rules:
- Output valid JSON only, without markdown wrappers.
- Every H block is its own unit. NEVER merge two headings into one unit, however
  similar their names or their text. Two rooms that differ only by a number are
  two units.
- `name`: the unit's product name, copied from its heading, with a bare instance
  identifier removed -- a room number, a pair of room numbers, an inventory
  count. Keep every word that describes the product itself, including what makes
  it accessible, double, air-conditioned or equipped.
    "חדר צוות מאובזר ומונגש חדר מספר 5"  -> "חדר צוות מאובזר ומונגש"
    "חדרי צוות חדרים 3-4"                 -> "חדרי צוות"
    "בונגלו עם מזגן"                      -> "בונגלו עם מזגן"
  If removing the identifier would give two headings in this panel the same
  name, keep the identifier on both so they stay apart.
- `blocks`: the P block numbers that describe this unit. Usually one; sometimes
  an introduction and then a specification.
- `rules`: P block numbers stating something about every unit, or about booking
  in general -- minimum nights, how a room is assigned, what can be ordered at
  extra cost. A paragraph listing what one room contains is NOT a rule.
- A paragraph belongs either to one unit or to `rules`, never to both, and every
  P block must appear exactly once.
- `notes`: anything a later reader would need and cannot see from one unit alone.
  Write one when two units in this panel are near-identical -- say which they are
  and what actually distinguishes them. Empty list when there is nothing to say.
- Use no quotation marks of any kind inside a string value, and keep each note to
  one sentence. A quote mark inside a string breaks the JSON and the whole panel
  is lost.

Schema:
{
  "units": [{"heading": int, "name": str, "blocks": [int]}],
  "rules": [int],
  "notes": [{"units": [str], "note": str}]
}
"""


@dataclass
class SegmentedPanel:
    """What the model made of one panel."""

    units: list[LodgingUnit]
    rules: list[str]
    notes: list[dict]


def render_blocks(blocks: list[Block]) -> str:
    """The panel as the numbered list the segmenter is shown."""
    kinds = {"heading": "H", "para": "P", "scope": "S"}
    lines = []
    for block in blocks:
        count = f" ({block.unit_count})" if block.unit_count is not None else ""
        lines.append(f"{block.index} {kinds[block.kind]}:{count} {block.text}")
    return "\n".join(lines)


class LodgingSegmenterLLMClient:
    """Which paragraph belongs to which unit, and what each unit is called.

    The panel separates units from rules structurally almost everywhere -- a
    heading claims the paragraph after it -- but not always: at Khan Be'erot one
    room has an introduction and then its specification, so its second paragraph
    is the unit's, while three lines later another room's second paragraph is the
    site's minimum-nights rule. Same markup, and the bold/regular formatting cuts
    across the distinction rather than along it, so it is a reading of the text.

    The model only labels and names. Paragraph text comes back by block number
    and is never rewritten, so an evidence span downstream is still verbatim.
    """

    MODEL = QWEN_INSTRUCT_30B_MODEL
    TEMPERATURE = 0
    # A backstop for the same failure `segment_panel` prevents: a labelling
    # reply is a few hundred tokens, so anything past this is a loop and should
    # fail loudly rather than be parsed.
    MAX_TOKENS = 4000

    def __init__(
        self,
        client: OpenAI | None = None,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._client = client
        self.model = model or self.MODEL
        self.system_prompt = system_prompt or SEGMENT_PROMPT

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = make_nebius_openai_client()
        return self._client

    def segment(
        self, blocks: list[Block], *, usage: LlmUsage | None = None
    ) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": render_blocks(blocks)},
            ],
            temperature=self.TEMPERATURE,
            max_tokens=self.MAX_TOKENS,
        )
        if usage is not None:
            usage.add_chat(response.usage, role="lodging_segment", model=self.model)
        raw = response.choices[0].message.content or ""
        try:
            return _parse_json_payload(raw)
        except ValueError as exc:
            # The failure is nearly always an unescaped quote inside a note, and
            # the whole panel is lost with it. Say which panel and show where.
            raise ValueError(f"segmenter JSON invalid ({exc}): {raw[:400]!r}") from exc


def segment_panel(
    blocks: list[Block],
    segmenter: LodgingSegmenterLLMClient,
    *,
    usage: LlmUsage | None = None,
) -> SegmentedPanel:
    """Units and rules for one panel, asking the model only when there is a
    question to answer.

    A panel with no paragraphs has nothing to attribute: every heading is a unit
    and there are no rules. Asking anyway loses the site -- Yehudia's whole panel
    is a single `<h4>`, and told to list the paragraph indices belonging to it
    when there are none, the 30B emitted a runaway `[1, 2, 3, ... 78 ...]` and
    ran past the end of the JSON.
    """
    if not any(b.kind == "para" for b in blocks):
        return assemble_units(blocks, {"units": [], "rules": [], "notes": []})
    return assemble_units(blocks, segmenter.segment(blocks, usage=usage))


def assemble_units(blocks: list[Block], answer: dict) -> SegmentedPanel:
    """The model's labelling, with the never-merge rule enforced over it.

    One heading is one unit, always. If two headings in a panel come back with
    the same name, both keep their heading verbatim instead -- a merge would
    cost a row, and at Metsada the two colliding rooms sleep 7 and 5.
    """
    by_index = {b.index: b for b in blocks}
    headings = [b for b in blocks if b.kind == "heading"]
    named: dict[int, str] = {}
    texts: dict[int, list[str]] = {h.index: [] for h in headings}
    claimed: set[int] = set()

    for item in answer.get("units") or []:
        head = by_index.get(item.get("heading"))
        if head is None or head.kind != "heading":
            continue
        name = fold(str(item.get("name") or "")) or head.text
        named[head.index] = name
        for number in item.get("blocks") or []:
            block = by_index.get(number)
            if block is not None and block.kind == "para":
                texts[head.index].append(block.text)
                claimed.add(block.index)

    # A heading the model skipped is still a unit; it keeps its own name.
    for head in headings:
        named.setdefault(head.index, head.text)

    # Never merge: a duplicated name sends every colliding heading back to its
    # verbatim text, which is unique by construction.
    collisions = {n for n in named.values() if list(named.values()).count(n) > 1}
    for index, name in list(named.items()):
        if name in collisions:
            named[index] = by_index[index].text

    units = [
        LodgingUnit(
            named[h.index], h.unit_count, "\n".join(texts[h.index]).strip(), h.scope
        )
        for h in headings
    ]
    rules = [
        by_index[n].text
        for n in (answer.get("rules") or [])
        if n in by_index and by_index[n].kind == "para" and n not in claimed
    ]
    return SegmentedPanel(units, rules, list(answer.get("notes") or []))

