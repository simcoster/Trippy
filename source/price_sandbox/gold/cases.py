"""Gold quote cases keyed by a unique fragment of the parks.org.il URL."""

from __future__ import annotations

TENT = "לינת שטח באוהלים פרטיים"


def _params(lodging: str, **kwargs) -> dict:
    payload = {"lodging": lodging, "adults_num": 1}
    payload.update(kwargs)
    return payload


def _tent_band(
    match: str,
    *,
    adult: float,
    child: float,
    matmon_adult: float,
    soldier: float,
    senior: float,
    extra: list[dict] | None = None,
) -> dict:
    cases = [
        {
            "note": "two adults weekday tent",
            "params": _params(TENT, adults_num=2),
            "expected_price": adult * 2,
        },
        {
            "note": "adults plus child and toddler",
            "params": _params(TENT, adults_num=2, child_ages=[7, 4]),
            "expected_price": adult * 2 + child,
        },
        {
            "note": "two adults matmon",
            "params": _params(TENT, adults_num=2, is_matmon_sub=True),
            "expected_price": matmon_adult * 2,
        },
        {
            "note": "one soldier",
            "params": _params(TENT, adults_num=1, is_soldier=True),
            "expected_price": soldier,
        },
        {
            "note": "one senior",
            "params": _params(TENT, adults_num=1, is_senior=True),
            "expected_price": senior,
        },
    ]
    if extra:
        cases = extra
    return {"match": match, "cases": cases}


CATALOG: list[dict] = [
    {
        "match": "חורשת-טל",
        "cases": [
            {
                "note": "two adults weekday tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 152.0,
            },
            {
                "note": "two adults two children one toddler",
                "params": _params(TENT, adults_num=2, child_ages=[5, 7, 4]),
                "expected_price": 268.0,
            },
            {
                "note": "two adults matmon tent",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 114.0,
            },
            {
                "note": "bungalow weekday unit",
                "params": _params("בונגלו עם מזגן", adults_num=2),
                "expected_price": 430.0,
            },
            {
                "note": "bungalow weekend late checkout",
                "params": _params(
                    "בונגלו עם מזגן",
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                "expected_price": 795.0,
            },
        ],
    },
    {
        "match": "אכזיב",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 152.0,
            },
            {
                "note": "tent with child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[8, 4]),
                "expected_price": 210.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 114.0,
            },
            {
                "note": "husha weekday",
                "params": _params("חושה", adults_num=2),
                "expected_price": 350.0,
            },
            {
                "note": "husha weekend late checkout",
                "params": _params(
                    "חושה",
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                "expected_price": 675.0,
            },
        ],
    },
    {
        "match": "יחיעם",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 128.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 175.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 96.0,
            },
            {
                "note": "pitch included occupancy",
                "params": _params("מתחם pitch", adults_num=4),
                "expected_price": 430.0,
            },
            {
                "note": "pitch one extra adult",
                "params": _params("מתחם pitch", adults_num=5),
                "expected_price": 494.0,
            },
        ],
    },
    _tent_band(
        "נחל-עמוד",
        adult=47.0,
        child=35.0,
        matmon_adult=35.0,
        soldier=35.0,
        senior=24.0,
        extra=[
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 94.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 129.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 70.0,
            },
            {
                "note": "family tent included 4",
                "params": _params("השכרת אוהל קמפינג משפחתי", adults_num=4),
                "expected_price": 292.0,
            },
            {
                "note": "family tent fifth person",
                "params": _params("השכרת אוהל קמפינג משפחתי", adults_num=5),
                "expected_price": 365.0,
            },
        ],
    ),
    _tent_band("יהודיה", adult=64.0, child=47.0, matmon_adult=48.0, soldier=47.0, senior=32.0),
    {
        "match": "משמר-הכרמל",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 128.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 175.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 96.0,
            },
            {
                "note": "staff room weekday",
                "params": _params("חדר צוות קטן", adults_num=2),
                "expected_price": 430.0,
            },
            {
                "note": "staff room weekend late checkout",
                "params": _params(
                    "חדר צוות קטן",
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                "expected_price": 795.0,
            },
        ],
    },
    {
        "match": "כוכב-הירדן",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 94.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 129.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 70.0,
            },
            {
                "note": "family tent included 4",
                "params": _params("השכרת אוהל קמפינג משפחתי", adults_num=4),
                "expected_price": 292.0,
            },
            {
                "note": "one soldier",
                "params": _params(TENT, adults_num=1, is_soldier=True),
                "expected_price": 35.0,
            },
        ],
    },
    {
        "match": "מעיין-חרוד",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 152.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 210.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 114.0,
            },
            {
                "note": "pitch included 4",
                "params": _params("מתחם pitch", adults_num=4),
                "expected_price": 476.0,
            },
            {
                "note": "caravan three adults",
                "params": _params("עמדת חניה לקרוואן", adults_num=3),
                "expected_price": 381.0,
            },
        ],
    },
    _tent_band("גן-השלושה", adult=76.0, child=58.0, matmon_adult=57.0, soldier=58.0, senior=38.0),
    {
        "match": "ירקון",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 128.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 175.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 96.0,
            },
            {
                "note": "family tent included 4",
                "params": _params("השכרת אוהל קמפינג משפחתי", adults_num=4),
                "expected_price": 350.0,
            },
            {
                "note": "caravan three adults",
                "params": _params("עמדת חניה לקרוואן", adults_num=3),
                "expected_price": 294.0,
            },
        ],
    },
    _tent_band("הקסטל", adult=64.0, child=47.0, matmon_adult=48.0, soldier=47.0, senior=32.0),
    _tent_band("אשקלון", adult=64.0, child=47.0, matmon_adult=48.0, soldier=47.0, senior=32.0),
    {
        "match": "הבשור",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 128.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 175.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 96.0,
            },
            {
                "note": "fixed mahal included 10",
                "params": _params("מאהל גדול קבוע", adults_num=10),
                "expected_price": 860.0,
            },
            {
                "note": "caravan three adults",
                "params": _params("עמדת חניה לקרוואן", adults_num=3),
                "expected_price": 344.0,
            },
        ],
    },
    {
        "match": "מצדה",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 128.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 175.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 96.0,
            },
            {
                "note": "family tent included 4",
                "params": _params("השכרת אוהל קמפינג משפחתי", adults_num=4),
                "expected_price": 350.0,
            },
            {
                "note": "staff room weekday",
                "params": _params("חדר צוות גדול", adults_num=2),
                "expected_price": 480.0,
            },
        ],
    },
    _tent_band("תל-ערד", adult=64.0, child=47.0, matmon_adult=48.0, soldier=47.0, senior=32.0),
    _tent_band("ממשית", adult=64.0, child=47.0, matmon_adult=48.0, soldier=47.0, senior=32.0),
    {
        "match": "בארות",
        "cases": [
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 128.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 175.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 96.0,
            },
            {
                "note": "small staff weekday",
                "params": _params("חדר צוות קטן", adults_num=2),
                "expected_price": 430.0,
            },
            {
                "note": "small staff weekend late checkout",
                "params": _params(
                    "חדר צוות קטן",
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                "expected_price": 795.0,
            },
        ],
    },
    _tent_band("יוטבתה", adult=64.0, child=47.0, matmon_adult=48.0, soldier=47.0, senior=32.0, extra=[
            {
                "note": "two adults tent",
                "params": _params(TENT, adults_num=2),
                "expected_price": 128.0,
            },
            {
                "note": "tent child and toddler",
                "params": _params(TENT, adults_num=2, child_ages=[6, 3]),
                "expected_price": 175.0,
            },
            {
                "note": "two adults matmon",
                "params": _params(TENT, adults_num=2, is_matmon_sub=True),
                "expected_price": 96.0,
            },
            {
                "note": "family tent included 4",
                "params": _params("השכרת אוהל קמפינג משפחתי", adults_num=4),
                "expected_price": 350.0,
            },
            {
                "note": "couple tent included 2",
                "params": _params("השכרת אוהל קמפינג זוגי", adults_num=2),
                "expected_price": 192.0,
            },
        ]),
]
