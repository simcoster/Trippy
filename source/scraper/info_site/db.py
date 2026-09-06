"""Postgres writes for info-site list prices.

`info_website_names` rows are created by `scrape-rooms` off the lodging panel,
which is the campsite's catalog. This scrape *finds* the row a rate-card label
belongs to and attaches a price to it, the same way the availability scrape
finds a type for a booking name. A label matching nothing is skipped and
reported rather than inventing a lodging product the operator never listed.
"""

from __future__ import annotations

from collections import defaultdict

from source.scraper.amenity_enrichment.llm import LlmUsage
from source.scraper.info_site.match_listing import (
    InfoWebsiteNameMatcher,
    match_info_website_name,
    rescue_info_website_names,
)

from .schemas import ClassifiedPriceRow

# Below this the match is worth a person's eye. The matcher is told never to
# refuse -- a rate card and a lodging panel describe one campsite, so a price
# belongs somewhere -- so a poor match reports itself as a low number instead.
UNCERTAIN_BELOW = 0.7

LOAD_INFO_WEBSITE_NAMES_SQL = """
SELECT id, name FROM info_website_names WHERE site_id = %(site_id)s ORDER BY id
"""

DELETE_REGULAR_LIST_PRICES_SQL = """
DELETE FROM list_prices
WHERE site_id = %(site_id)s
  AND rate_class = %(rate_class)s
"""

INSERT_LIST_PRICE_SQL = """
INSERT INTO list_prices (
    site_id, info_website_name_id, guest_type, rate_period, rate_class,
    price, currency, notes, raw_label
) VALUES (
    %(site_id)s, %(info_website_name_id)s, %(guest_type)s, %(rate_period)s,
    %(rate_class)s, %(price)s, %(currency)s, %(notes)s, %(raw_label)s
)
ON CONFLICT ON CONSTRAINT list_prices_unique_rate DO UPDATE
SET price = EXCLUDED.price,
    currency = EXCLUDED.currency,
    notes = EXCLUDED.notes,
    raw_label = EXCLUDED.raw_label,
    scraped_at = now(),
    updated_at = now()
RETURNING id;
"""

FILL_BOOKING_HOTEL_ID_SQL = """
UPDATE campsites
SET booking_hotel_id = %(booking_hotel_id)s
WHERE id = %(id)s
  AND booking_hotel_id IS NULL
RETURNING id, booking_hotel_id;
"""


CREATE_INFO_WEBSITE_NAME_SQL = """
INSERT INTO info_website_names (site_id, name)
VALUES (%(site_id)s, %(name)s)
ON CONFLICT (site_id, name) DO UPDATE SET name = EXCLUDED.name
RETURNING id
"""


def get_or_create_info_website_name(cur, *, site_id: int, name: str) -> int:
    """The lodging product row for a name, created if new.

    Called by `scrape-rooms` off the lodging panel headings, and by nothing
    else: the panel is the catalog, so a name that is not on it is not a
    product. The price scrape matches against what this created.
    """
    cur.execute(
        CREATE_INFO_WEBSITE_NAME_SQL,
        {"site_id": site_id, "name": name},
    )
    return int(cur.fetchone()[0])


def maybe_fill_booking_hotel_id(
    cur, *, site_id: int, booking_hotel_id: str | None
) -> str | None:
    if not booking_hotel_id:
        return None
    cur.execute(
        FILL_BOOKING_HOTEL_ID_SQL,
        {"id": site_id, "booking_hotel_id": booking_hotel_id},
    )
    row = cur.fetchone()
    return row[1] if row else None


def colliding_rows(
    resolutions: list[tuple[ClassifiedPriceRow, list[int], float | None]],
    rate_class: str,
) -> list[list[int]]:
    """Indices of rows that would overwrite each other, worst pair first.

    `list_prices_unique_rate` is (product, guest type, rate period, class), so
    two rate lines resolving to the same product do not both survive the insert
    -- the second wins and the first is gone with nothing printed. This finds
    that before any of it is written, and it needs no model and no confidence:
    the rate card priced them on separate lines, so they are separate products
    and one of the two matches is simply wrong.
    """
    groups: dict[tuple, list[int]] = defaultdict(list)
    for index, (row, name_ids, _confidence) in enumerate(resolutions):
        for name_id in name_ids:
            key = (name_id, row.guest_type, row.rate_period, rate_class)
            groups[key].append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def resolve_collisions(
    resolutions: list[tuple[ClassifiedPriceRow, list[int], float | None]],
    names: list[tuple[int, str]],
    *,
    rate_class: str,
    matcher: InfoWebsiteNameMatcher,
    usage: LlmUsage | None = None,
    unmatched_sink: list[str] | None = None,
) -> None:
    """Give each of two colliding rate lines a product of its own, in place.

    Only a pair is asked about: three lines on one product is a different shape
    -- likely a catalog missing an entry rather than one bad match -- and is
    reported instead of guessed at.
    """
    by_id = dict(names)
    for indices in colliding_rows(resolutions, rate_class):
        first_row, first_ids, _ = resolutions[indices[0]]
        if len(indices) != 2:
            labels = ", ".join(
                repr(resolutions[i][0].raw_label) for i in indices
            )
            print(f"      {len(indices)} labels on one product, left as is: {labels}")
            if unmatched_sink is not None:
                unmatched_sink.append(f"{len(indices)}-way collision: {labels}")
            continue
        second_row, second_ids, _ = resolutions[indices[1]]
        collided_on = by_id.get(first_ids[0], "")
        first, second, confidence = matcher.pick_pair(
            first_row.raw_label,
            second_row.raw_label,
            collided_on,
            [name for _, name in names],
            usage=usage,
        )
        if first is None or second is None:
            print(
                f"      COLLISION unresolved on {collided_on!r}: "
                f"{first_row.raw_label!r} / {second_row.raw_label!r}"
            )
            if unmatched_sink is not None:
                unmatched_sink.append(
                    f"unresolved collision on {collided_on}: "
                    f"{first_row.raw_label} / {second_row.raw_label}"
                )
            continue
        ids = {name: row_id for row_id, name in names}
        resolutions[indices[0]] = (first_row, [ids[first]], confidence)
        resolutions[indices[1]] = (second_row, [ids[second]], confidence)
        print(
            f"      COLLISION on {collided_on!r} split: "
            f"{first_row.raw_label!r} -> {first!r}, "
            f"{second_row.raw_label!r} -> {second!r}"
        )


def snapshot_list_prices(
    conn,
    *,
    site_id: int,
    rows: list[ClassifiedPriceRow],
    rate_class: str = "regular",
    currency: str = "ILS",
    matcher: InfoWebsiteNameMatcher | None = None,
    usage: LlmUsage | None = None,
    unmatched_sink: list[str] | None = None,
) -> list[ClassifiedPriceRow]:
    """Replace regular list prices for a site. Persists lodging rows only.

    Each rate-card label is resolved to an existing `info_website_names` row --
    exact name on the classified type, else one 235B pick shown the full
    rate-card label, over the names `scrape-rooms` created. A pick the model is
    unsure of gets a second call that may name several products, and the price
    is then filed against each of them. Two labels landing on one product is
    settled last, once every row is resolved -- see `resolve_collisions`. A label
    that resolves to nothing is skipped: the rate card sometimes prices things
    the lodging panel does not list, and a price with no product is not
    something the planner can use.
    """
    from .classify import lodging_rows_to_persist

    lodging = lodging_rows_to_persist(rows)
    with conn.cursor() as cur:
        cur.execute(LOAD_INFO_WEBSITE_NAMES_SQL, {"site_id": site_id})
        names = [(int(r[0]), r[1]) for r in cur.fetchall()]
        cur.execute(
            DELETE_REGULAR_LIST_PRICES_SQL,
            {"site_id": site_id, "rate_class": rate_class},
        )
        stored: list[ClassifiedPriceRow] = []
        resolutions: list[tuple[ClassifiedPriceRow, list[int], float | None]] = []
        for row in lodging:
            name_id, confidence = match_info_website_name(
                row.accommodation_type,
                names,
                # The label as the rate card wrote it. The classifier's
                # normalised type is what can match a catalog name exactly, but
                # it is also what drops the room numbers, and the model needs
                # them: `חדר צוות גדול` cannot be told from three other staff
                # rooms, `... (חדרים 5 ו-6)` can.
                full_label=row.raw_label,
                matcher=matcher,
                usage=usage,
            )
            name_ids = [] if name_id is None else [name_id]
            doubted = name_id is None or (
                confidence is not None and confidence < UNCERTAIN_BELOW
            )
            if doubted and matcher is not None:
                # A rate card sometimes prices two products on one line, and a
                # single pick has to be wrong about one of them. Only a doubted
                # answer is worth a second call, and only a doubted one is
                # allowed to come back with several names.
                rescued, rescued_confidence = rescue_info_website_names(
                    row.raw_label, names, matcher=matcher, usage=usage
                )
                if rescued:
                    name_ids, confidence = rescued, rescued_confidence
                    if len(rescued) > 1:
                        print(
                            f"      SPLIT across {len(rescued)}: {row.raw_label!r}"
                        )
            if not name_ids:
                # The prompt forbids a refusal, so this is the model failing to
                # follow it rather than a label with no home. Attach the price
                # to the first candidate at confidence 0 and say so: a price
                # filed against the wrong unit is visible and fixable, a price
                # dropped on the floor is neither.
                name_ids, confidence = [names[0][0]], 0.0
                print(f"      FORCED MATCH (model refused): {row.raw_label!r}")
            if confidence is not None and confidence < UNCERTAIN_BELOW:
                picked = ", ".join(
                    n for i, n in names if i in name_ids
                )
                print(
                    f"      UNCERTAIN {confidence:.2f}: "
                    f"{row.raw_label!r} -> {picked!r}"
                )
                if unmatched_sink is not None:
                    unmatched_sink.append(
                        f"{row.raw_label} -> {picked} ({confidence:.2f})"
                    )
            resolutions.append((row, name_ids, confidence))

        # Every row is resolved before any is written: a clash is only visible
        # once both halves of it exist, and it has to be settled before the
        # insert rather than discovered from a row count afterwards.
        if matcher is not None:
            resolve_collisions(
                resolutions,
                names,
                rate_class=rate_class,
                matcher=matcher,
                usage=usage,
                unmatched_sink=unmatched_sink,
            )

        for row, name_ids, _confidence in resolutions:
            stored.append(row)
            for name_id in name_ids:
                cur.execute(
                    INSERT_LIST_PRICE_SQL,
                    {
                        "site_id": site_id,
                        "info_website_name_id": name_id,
                        "guest_type": row.guest_type,
                        "rate_period": row.rate_period,
                        "rate_class": rate_class,
                        "price": row.price,
                        "currency": currency,
                        "notes": row.notes,
                        "raw_label": row.raw_label,
                    },
                )
    return stored
