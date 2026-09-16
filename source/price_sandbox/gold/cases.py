"""Gold quote cases keyed by a unique fragment of the parks.org.il URL."""

from __future__ import annotations

TENT = "לינת שטח באוהלים פרטיים"
REGULAR = "רגיל"
MATMON = "מנוי"
SOLDIER = "חייל בשירות חובה + שירות לאומי"
MILUIM = "משרת מילואים פעיל"
SENIOR = "אזרח ותיק"
STUDENT = "סטודנט"
DISABLED = 'נכה צה"ל ומלווה'
IDENTITY_GUEST_TYPES = (
    REGULAR,
    MATMON,
    SOLDIER,
    MILUIM,
    SENIOR,
    STUDENT,
    DISABLED,
)

HUSHA = "חושה"
HUSHA_DOUBLE = "חושה כפולה"
HUSHA_AC = "חושה עם מזגן שירותים ומקלחת"
HUSHA_DOUBLE_AC = "חושה כפולה עם מזגן שירותים ומקלחת"
BUNGALOW = "בונגלו עם מזגן"
STAFF = "חדר צוות"
STAFF_WOOD = "חדר צוות עץ"
CARAVAN = "עמדת חניה לקרוואן"
PITCH = "מתחם pitch"
FAMILY_TENT = "השכרת אוהל קמפינג משפחתי"
COUPLE_TENT = "השכרת אוהל קמפינג זוגי"
SMALL_STAFF = "חדר צוות קטן"
LARGE_STAFF = "חדר צוות גדול"
FIXED_MAHAL = "מאהל גדול קבוע"


def _ils(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _params(lodging: str, **kwargs) -> dict:
    payload = {
        "lodging": lodging,
        "adults_num": 1,
        "child_num": 0,
        "guest_type": REGULAR,
    }
    payload.update(kwargs)
    if "child_num" not in kwargs:
        payload["child_num"] = len(payload.get("child_ages") or [])
    return payload


def _case(note: str, params: dict, price: float, explanation: str) -> dict:
    return {
        "note": note,
        "params": params,
        "expected_price": price,
        "explanation": explanation,
    }


def _identity_adult(note: str, guest_type: str, price: float, label: str) -> dict:
    return _case(
        note,
        _params(TENT, adults_num=1, guest_type=guest_type),
        price,
        f"1 {label} [{_ils(price)}]",
    )


def _soldier(price: float) -> dict:
    return _identity_adult("one soldier", SOLDIER, price, "soldier")


def _senior(price: float) -> dict:
    return _identity_adult("one senior", SENIOR, price, "senior")


def _unit_weekday(note: str, lodging: str, price: float, label: str) -> dict:
    return _case(
        note,
        _params(lodging, adults_num=2),
        price,
        f"{label} weekday unit [{_ils(price)}] (party size ignored)",
    )


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
        _case(
            "two adults weekday tent",
            _params(TENT, adults_num=2),
            adult * 2,
            f"2 adults [{_ils(adult)}]",
        ),
        _case(
            "adults plus child and toddler",
            _params(TENT, adults_num=2, child_ages=[7, 4]),
            adult * 2 + child,
            f"2 adults [{_ils(adult)}] + 1 child [{_ils(child)}] (age 7) "
            f"+ 1 toddler [free] (age 4)",
        ),
        _case(
            "two adults matmon",
            _params(TENT, adults_num=2, guest_type=MATMON),
            matmon_adult * 2,
            f"2 adults [{_ils(matmon_adult)}]; Matmon",
        ),
        _soldier(soldier),
        _senior(senior),
    ]
    if extra:
        cases = cases + extra
    return {"match": match, "cases": cases}


CATALOG: list[dict] = [
    {
        "match": "חורשת-טל",
        "cases": [
            _case(
                "two adults weekday tent",
                _params(TENT, adults_num=2),
                152.0,
                "2 adults [76]",
            ),
            _case(
                "two adults two children one toddler",
                _params(TENT, adults_num=2, child_ages=[5, 7, 4]),
                268.0,
                "2 adults [76] + 2 children [58] [ages 5,7] + 1 toddler [free] (age 4)",
            ),
            _case(
                "two adults matmon tent",
                _params(TENT, adults_num=2, guest_type=MATMON),
                114.0,
                "2 adults [57]; Matmon",
            ),
            _soldier(58.0),
            _senior(38.0),
            _case(
                "bungalow weekday unit",
                _params(BUNGALOW, adults_num=2),
                430.0,
                "bungalow weekday unit [430] (party size ignored)",
            ),
            _case(
                "bungalow weekend late checkout",
                _params(
                    BUNGALOW,
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                795.0,
                "bungalow weekend unit [530] + late checkout [265] (exit 13:00)",
            ),
            _unit_weekday("staff room weekday", STAFF, 480.0, "staff room"),
            _unit_weekday("wood staff weekday", STAFF_WOOD, 730.0, "wood staff room"),
            _case(
                "caravan two adults included",
                _params(CARAVAN, adults_num=2),
                305.0,
                "caravan bay [305] includes up to 2 guests",
            ),
        ],
    },
    {
        "match": "אכזיב",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                152.0,
                "2 adults [76]",
            ),
            _case(
                "tent with child and toddler",
                _params(TENT, adults_num=2, child_ages=[8, 4]),
                210.0,
                "2 adults [76] + 1 child [58] (age 8) + 1 toddler [free] (age 4)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                114.0,
                "2 adults [57]; Matmon",
            ),
            _soldier(58.0),
            _identity_adult("one miluim", MILUIM, 65.0, "miluim"),
            _senior(38.0),
            _identity_adult("one student", STUDENT, 65.0, "student"),
            _identity_adult("one disabled", DISABLED, 38.0, "disabled"),
            _case(
                "tent group 30 adults",
                _params(TENT, adults_num=30),
                1950.0,
                "30 adults [65] group occupancy override (min 30)",
            ),
            _case(
                "tent group 30 matmon adults",
                _params(TENT, adults_num=30, guest_type=MATMON),
                1950.0,
                "30 adults [65] group occupancy override (min 30); Matmon ignored",
            ),
            _unit_weekday("husha weekday", HUSHA, 350.0, "husha"),
            _case(
                "husha weekend late checkout",
                _params(
                    HUSHA,
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                675.0,
                "husha weekend unit [450] + late checkout [225] (exit 13:00)",
            ),
            _unit_weekday(
                "double husha weekday", HUSHA_DOUBLE, 700.0, "double husha"
            ),
            _unit_weekday("ac husha weekday", HUSHA_AC, 430.0, "ac husha"),
            _unit_weekday(
                "double ac husha weekday",
                HUSHA_DOUBLE_AC,
                780.0,
                "double ac husha",
            ),
        ],
    },
    {
        "match": "יחיעם",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                128.0,
                "2 adults [64]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                175.0,
                "2 adults [64] + 1 child [47] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                96.0,
                "2 adults [48]; Matmon",
            ),
            _soldier(47.0),
            _senior(32.0),
            _case(
                "pitch included occupancy",
                _params(PITCH, adults_num=4),
                430.0,
                "pitch unit [430] includes up to 4 guests",
            ),
            _case(
                "pitch one extra adult",
                _params(PITCH, adults_num=5),
                494.0,
                "pitch unit [430] includes 4 + 1 extra adult [64]",
            ),
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
            _case(
                "family tent included 4",
                _params(FAMILY_TENT, adults_num=4),
                292.0,
                "family tent [292] includes up to 4 guests",
            ),
            _case(
                "family tent fifth person",
                _params(FAMILY_TENT, adults_num=5),
                365.0,
                "family tent [292] includes 4 + 1 extra person [73]",
            ),
        ],
    ),
    _tent_band(
        "יהודיה",
        adult=64.0,
        child=47.0,
        matmon_adult=48.0,
        soldier=47.0,
        senior=32.0,
    ),
    {
        "match": "משמר-הכרמל",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                128.0,
                "2 adults [64]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                175.0,
                "2 adults [64] + 1 child [47] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                96.0,
                "2 adults [48]; Matmon",
            ),
            _soldier(47.0),
            _senior(32.0),
            _case(
                "staff room weekday",
                _params(SMALL_STAFF, adults_num=2),
                430.0,
                "small staff room weekday unit [430] (party size ignored)",
            ),
            _case(
                "staff room weekend late checkout",
                _params(
                    SMALL_STAFF,
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                795.0,
                "small staff room weekend unit [530] + late checkout [265] (exit 13:00)",
            ),
        ],
    },
    {
        "match": "כוכב-הירדן",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                94.0,
                "2 adults [47]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                129.0,
                "2 adults [47] + 1 child [35] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                70.0,
                "2 adults [35]; Matmon",
            ),
            _soldier(35.0),
            _senior(24.0),
            _case(
                "family tent included 4",
                _params(FAMILY_TENT, adults_num=4),
                292.0,
                "family tent [292] includes up to 4 guests",
            ),
        ],
    },
    {
        "match": "מעיין-חרוד",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                152.0,
                "2 adults [76]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                210.0,
                "2 adults [76] + 1 child [58] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                114.0,
                "2 adults [57]; Matmon",
            ),
            _soldier(58.0),
            _senior(38.0),
            _case(
                "pitch included 4",
                _params(PITCH, adults_num=4),
                476.0,
                "pitch unit [476] includes up to 4 guests",
            ),
            _case(
                "caravan three adults",
                _params(CARAVAN, adults_num=3),
                381.0,
                "caravan bay [305] includes 2 + 1 extra adult [76]",
            ),
        ],
    },
    _tent_band(
        "גן-השלושה",
        adult=76.0,
        child=58.0,
        matmon_adult=57.0,
        soldier=58.0,
        senior=38.0,
    ),
    {
        "match": "ירקון",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                128.0,
                "2 adults [64]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                175.0,
                "2 adults [64] + 1 child [47] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                96.0,
                "2 adults [48]; Matmon",
            ),
            _soldier(47.0),
            _senior(32.0),
            _case(
                "family tent included 4",
                _params(FAMILY_TENT, adults_num=4),
                350.0,
                "family tent [350] includes up to 4 guests",
            ),
            _case(
                "caravan three adults",
                _params(CARAVAN, adults_num=3),
                294.0,
                "caravan bay [230] includes 2 + 1 extra adult [64]",
            ),
        ],
    },
    _tent_band(
        "הקסטל",
        adult=64.0,
        child=47.0,
        matmon_adult=48.0,
        soldier=47.0,
        senior=32.0,
    ),
    _tent_band(
        "אשקלון",
        adult=64.0,
        child=47.0,
        matmon_adult=48.0,
        soldier=47.0,
        senior=32.0,
    ),
    {
        "match": "הבשור",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                128.0,
                "2 adults [64]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                175.0,
                "2 adults [64] + 1 child [47] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                96.0,
                "2 adults [48]; Matmon",
            ),
            _soldier(47.0),
            _senior(32.0),
            _case(
                "fixed mahal included 10",
                _params(FIXED_MAHAL, adults_num=10),
                860.0,
                "fixed mahal [860] includes up to 10 guests",
            ),
            _case(
                "caravan three adults",
                _params(CARAVAN, adults_num=3),
                344.0,
                "caravan bay [280] includes 2 + 1 extra adult [64]",
            ),
        ],
    },
    {
        "match": "מצדה",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                128.0,
                "2 adults [64]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                175.0,
                "2 adults [64] + 1 child [47] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                96.0,
                "2 adults [48]; Matmon",
            ),
            _soldier(47.0),
            _senior(32.0),
            _case(
                "family tent included 4",
                _params(FAMILY_TENT, adults_num=4),
                350.0,
                "family tent [350] includes up to 4 guests",
            ),
            _case(
                "staff room weekday",
                _params(LARGE_STAFF, adults_num=2),
                480.0,
                "large staff room weekday unit [480] (party size ignored)",
            ),
        ],
    },
    _tent_band(
        "תל-ערד",
        adult=64.0,
        child=47.0,
        matmon_adult=48.0,
        soldier=47.0,
        senior=32.0,
    ),
    _tent_band(
        "ממשית",
        adult=64.0,
        child=47.0,
        matmon_adult=48.0,
        soldier=47.0,
        senior=32.0,
    ),
    {
        "match": "בארות",
        "cases": [
            _case(
                "two adults tent",
                _params(TENT, adults_num=2),
                128.0,
                "2 adults [64]",
            ),
            _case(
                "tent child and toddler",
                _params(TENT, adults_num=2, child_ages=[6, 3]),
                175.0,
                "2 adults [64] + 1 child [47] (age 6) + 1 toddler [free] (age 3)",
            ),
            _case(
                "two adults matmon",
                _params(TENT, adults_num=2, guest_type=MATMON),
                96.0,
                "2 adults [48]; Matmon",
            ),
            _soldier(47.0),
            _senior(32.0),
            _case(
                "small staff weekday",
                _params(SMALL_STAFF, adults_num=2),
                430.0,
                "small staff room weekday unit [430] (party size ignored)",
            ),
            _case(
                "small staff weekend late checkout",
                _params(
                    SMALL_STAFF,
                    adults_num=2,
                    is_weekend_or_holiday=True,
                    planned_exit_time="13:00",
                ),
                795.0,
                "small staff room weekend unit [530] + late checkout [265] (exit 13:00)",
            ),
        ],
    },
    _tent_band(
        "יוטבתה",
        adult=64.0,
        child=47.0,
        matmon_adult=48.0,
        soldier=47.0,
        senior=32.0,
        extra=[
            _case(
                "family tent included 4",
                _params(FAMILY_TENT, adults_num=4),
                350.0,
                "family tent [350] includes up to 4 guests",
            ),
            _case(
                "couple tent included 2",
                _params(COUPLE_TENT, adults_num=2),
                192.0,
                "couple tent [192] includes up to 2 guests",
            ),
        ],
    ),
]
