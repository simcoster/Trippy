"""Public Streamlit chrome: language, copy, premade searches, page CSS."""

from __future__ import annotations

import base64
import random
from pathlib import Path
from typing import Literal

Lang = Literal["en", "he"]

GITHUB_URL = "https://github.com/simcoster/Trippy"
_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_BACKGROUND = _ASSETS / "background.png"

# Official GitHub mark (Simple Icons / GitHub logos, public domain).
GITHUB_MARK_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/></svg>"""

# Flags as data URIs so the page does not depend on emoji fonts.
_UK_FLAG = (
    "data:image/svg+xml;utf8,"
    "%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2060%2030%27%3E"
    "%3Cpath%20fill%3D%27%23012169%27%20d%3D%27M0%200h60v30H0z%27/%3E"
    "%3Cpath%20stroke%3D%27%23FFF%27%20stroke-width%3D%276%27%20d%3D%27m0%200%2060%2030m0-30L0%2030%27/%3E"
    "%3Cpath%20stroke%3D%27%23C8102E%27%20stroke-width%3D%274%27%20d%3D%27m0%200%2060%2030m0-30L0%2030%27/%3E"
    "%3Cpath%20stroke%3D%27%23FFF%27%20stroke-width%3D%2710%27%20d%3D%27M30%200v30M0%2015h60%27/%3E"
    "%3Cpath%20stroke%3D%27%23C8102E%27%20stroke-width%3D%276%27%20d%3D%27M30%200v30M0%2015h60%27/%3E"
    "%3C/svg%3E"
)
_IL_FLAG = (
    "data:image/svg+xml;utf8,"
    "%3Csvg%20xmlns%3D%27http%3A//www.w3.org/2000/svg%27%20viewBox%3D%270%200%2021%2015%27%3E"
    "%3Crect%20width%3D%2721%27%20height%3D%2715%27%20fill%3D%27%23fff%27/%3E"
    "%3Crect%20y%3D%272%27%20width%3D%2721%27%20height%3D%272%27%20fill%3D%27%230038b8%27/%3E"
    "%3Crect%20y%3D%2711%27%20width%3D%2721%27%20height%3D%272%27%20fill%3D%27%230038b8%27/%3E"
    "%3Cpath%20fill%3D%27none%27%20stroke%3D%27%230038b8%27%20stroke-width%3D%270.7%27%20"
    "d%3D%27M10.5%204.6%2012.8%208.6H8.2z%27/%3E"
    "%3Cpath%20fill%3D%27none%27%20stroke%3D%27%230038b8%27%20stroke-width%3D%270.7%27%20"
    "d%3D%27M10.5%2010.4%208.2%206.4h4.6z%27/%3E"
    "%3C/svg%3E"
)

# Five searches, each already written in both languages.
PREMADE_SEARCHES: tuple[tuple[str, str], ...] = (
    (
        "We're looking for a place for two adults next week between Tuesday and Thursday for one night, with at least two accessible toilets.",
        "אנחנו מחפשים מקום לשני מבוגרים בשבוע הבא בין שלישי לחמישי ללילה אחד, עם לפחות 2 שירותי נכים.",
    ),
    (
        "We're looking for a place for 2 adults and 2 kids for one night next week Tuesday-Thursday, with pools for the kids and a fridge, for up to 300 NIS.",
        "אנחנו מחפשים מקום לשני מבוגרים ושני ילדים ללילה אחד בשבוע הבא בין שלישי לחמישי, עם בריכות לילדים ומקרר, עד 300 ש״ח.",
    ),
    (
        "A couple this weekend near the sea, dogs allowed.",
        "זוג בסוף השבוע ליד הים, מותר כלבים.",
    ),
    (
        "Family of four next month in the desert. We need electricity and a fridge.",
        "משפחה של ארבעה בחודש הבא במדבר, צריכים חשמל ומקרר.",
    ),
    (
        "One night tonight, Shabbat observant, hot showers.",
        "לילה אחד הלילה, שומר שבת, מקלחות חמות.",
    ),
)

_COPY: dict[Lang, dict[str, str]] = {
    "en": {
        "title_public": "Trippy camping ⛺",
        "title_local": "Trippy camping ⛺ (local)",
        "ask_placeholder": "",
        "try_another": "Try another question!",
        "fill_random": "Fill random search",
        "something_wrong": "Something went wrong.",
        "quota_used": "You've used your questions.",
        "searches_left": "searches left",
        "question_left": "question left",
        "github": "GitHub",
        "readme_here": "Readme",
        "searching": "Searching",
        "ranking": "Ranking",
        "found_one": "Found 1 candidate, filtering",
        "found_many": "Found {n} candidates, filtering",
        "lang_en": "EN",
        "lang_he": "עב",
        "language": "Language",
    },
    "he": {
        "title_public": "Trippy camping ⛺",
        "title_local": "Trippy camping ⛺ (local)",
        "ask_placeholder": "",
        "try_another": "נסו שאלה אחרת!",
        "fill_random": "מילוי חיפוש אקראי",
        "something_wrong": "משהו השתבש.",
        "quota_used": "ניצלתם את השאלות.",
        "searches_left": "חיפושים נותרו",
        "question_left": "שאלה נותרה",
        "github": "github",
        "readme_here": "Readme",
        "searching": "מחפש",
        "ranking": "מדרג",
        "found_one": "נמצאה אפשרות אחת, מסנן",
        "found_many": "נמצאו {n} אפשרויות, מסנן",
        "lang_en": "EN",
        "lang_he": "עב",
        "language": "שפה",
    },
}


def copy(lang: Lang, key: str) -> str:
    return _COPY[lang][key]


def searches_left_label(remaining: int, lang: Lang) -> str:
    if remaining == 1:
        return copy(lang, "question_left")
    return copy(lang, "searches_left")


def localize_status(text: str, lang: Lang) -> str:
    if text == "Searching":
        return copy(lang, "searching")
    if text == "Ranking":
        return copy(lang, "ranking")
    if text.startswith("Found ") and text.endswith(" filtering"):
        try:
            n = int(text.split()[1])
        except (IndexError, ValueError):
            return text
        if n == 1:
            return copy(lang, "found_one")
        return copy(lang, "found_many").format(n=n)
    return text


def pick_premade(lang: Lang, last_index: int | None) -> tuple[int, str]:
    choices = [i for i in range(len(PREMADE_SEARCHES)) if i != last_index]
    if not choices:
        choices = list(range(len(PREMADE_SEARCHES)))
    index = random.choice(choices)
    en, he = PREMADE_SEARCHES[index]
    return index, he if lang == "he" else en


def search_in_lang(index: int, lang: Lang) -> str:
    en, he = PREMADE_SEARCHES[index]
    return he if lang == "he" else en


def background_data_uri() -> str:
    raw = _BACKGROUND.read_bytes()
    return "data:image/png;base64," + base64.standard_b64encode(raw).decode("ascii")


def dice_data_uri() -> str:
    raw = (_ASSETS / "dice.png").read_bytes()
    return "data:image/png;base64," + base64.standard_b64encode(raw).decode("ascii")


def page_css(background_uri: str, lang: Lang = "en") -> str:
    text_dir = "rtl" if lang == "he" else "ltr"
    dice_uri = dice_data_uri()
    return f"""
<style>
html, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
    direction: ltr;
}}
.trippy-questions-left,
[data-testid="stChatMessage"],
[data-testid="stBottom"] {{
    direction: {text_dir};
}}
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
#MainMenu {{
    display: none !important;
}}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li {{
    unicode-bidi: plaintext;
    text-align: start;
}}
.trippy-questions-left {{
    font-size: 1.15rem;
    line-height: 1.3;
    margin: 0 0 0.75rem 0;
    text-align: {"right" if lang == "he" else "left"};
    padding-right: {"14.5rem" if lang == "he" else "0"};
}}
.trippy-questions-left-n {{
    font-size: 1.45rem;
    font-weight: 700;
    color: #ff4b4b;
}}
.trippy-questions-left-label {{
    font-size: 1.85rem;
    font-weight: 650;
}}
.st-key-reset_chat button {{
    background-color: #21c354 !important;
    border-color: #21c354 !important;
}}
.st-key-reset_chat button:hover,
.st-key-reset_chat button:focus {{
    background-color: #1a9e43 !important;
    border-color: #1a9e43 !important;
    color: #fff !important;
}}
[data-testid="stSidebarContent"] {{
    display: flex;
    flex-direction: column;
}}
[data-testid="stSidebarUserContent"] {{
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    padding-bottom: 1rem !important;
}}
[data-testid="stSidebarUserContent"] > div {{
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
}}
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] {{
    flex: 1 1 auto;
}}
.st-key-github_readme {{
    margin-top: auto;
}}
.st-key-github_readme a {{
    min-height: 4.5rem;
    padding-top: 1.15rem;
    padding-bottom: 1.15rem;
    font-size: 1.2rem;
    font-weight: 650;
}}
[data-testid="stAppViewContainer"] {{
    background-color: transparent;
}}
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("{background_uri}");
    background-size: cover;
    background-position: center;
    opacity: 0.5;
    pointer-events: none;
    z-index: 0;
}}
[data-testid="stHeader"] {{
    background: transparent !important;
    pointer-events: none;
}}
[data-testid="stHeader"] button {{
    pointer-events: auto;
}}
[data-testid="stMain"],
[data-testid="stBottom"] {{
    position: relative;
    z-index: 1;
}}
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] [data-testid="stVerticalBlock"] {{
    background: transparent !important;
    box-shadow: none !important;
}}
[data-testid="stMainBlockContainer"] {{
    position: relative;
    padding-top: 0.4rem;
}}
[data-testid="stMainBlockContainer"] h1 {{
    direction: ltr;
    padding-right: 14.5rem;
    text-align: left;
}}
.st-key-lang_box {{
    position: absolute;
    top: 3.35rem;
    right: 8.6rem;
    width: auto !important;
    direction: ltr;
    z-index: 5;
}}
.st-key-github_mark {{
    position: absolute;
    top: 2.35rem;
    right: 0.4rem;
    width: auto !important;
    min-height: 7.2rem;
    direction: ltr;
    overflow: visible !important;
    z-index: 5;
}}
.st-key-github_mark [data-testid="stVerticalBlock"] {{
    display: flex !important;
    flex-direction: column !important;
    align-items: center;
    gap: 0.2rem;
    overflow: visible !important;
}}
.trippy-gh {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 4.4rem;
    height: 4.4rem;
    border-radius: 1rem;
    background: #111;
    color: #fff;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.28);
    text-decoration: none;
}}
.trippy-gh svg {{
    width: 2.7rem;
    height: 2.7rem;
}}
.trippy-gh:hover {{
    background: #000;
    color: #fff;
}}
.trippy-gh-caption {{
    display: block;
    font-size: 0.8rem;
    font-weight: 650;
    line-height: 1.15;
    white-space: nowrap;
    text-align: center;
    margin: 0;
}}
.trippy-gh-caption,
.trippy-gh-caption a {{
    color: #fff !important;
    text-decoration: none;
}}
.st-key-lang_box [data-testid="stButtonGroup"] {{
    direction: ltr !important;
    flex-direction: row !important;
}}
.st-key-lang_box [data-testid="stButtonGroup"] button {{
    font-size: 1.05rem;
    min-height: 2.7rem;
    padding: 0 0.7rem;
}}
.st-key-lang_box [data-testid="stButtonGroup"] button:nth-child(1)::before,
.st-key-lang_box [data-testid="stButtonGroup"] button:nth-child(2)::before {{
    content: "";
    display: inline-block;
    width: 1.55rem;
    height: 1.05rem;
    margin-inline-end: 0.4rem;
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(0, 0, 0, 0.15);
}}
.st-key-lang_box [data-testid="stButtonGroup"] button:nth-child(1)::before {{
    background-image: url("{_UK_FLAG}");
}}
.st-key-lang_box [data-testid="stButtonGroup"] button:nth-child(2)::before {{
    background-image: url("{_IL_FLAG}");
}}
[data-testid="stChatInput"] > div {{
    min-height: 6.2rem;
    align-items: stretch;
    background: rgba(255, 255, 255, 0.55) !important;
}}
[data-testid="stChatInputTextArea"],
[data-testid="stChatInput"] textarea {{
    min-height: 5.4rem !important;
    font-size: 1.8rem !important;
    line-height: 1.45 !important;
    padding-top: 0.85rem !important;
    padding-bottom: 0.85rem !important;
    padding-inline-end: 12.2rem !important;
    background: transparent !important;
    color: #111 !important;
    -webkit-text-fill-color: #111 !important;
    caret-color: #111 !important;
}}
[data-testid="stChatInputSubmitButton"],
[data-testid="stChatInput"] button {{
    width: 4.6rem !important;
    height: 4.6rem !important;
    min-width: 4.6rem !important;
    min-height: 4.6rem !important;
    padding: 0 !important;
}}
[data-testid="stChatInputSubmitButton"] svg,
[data-testid="stChatInput"] button svg {{
    width: 2.3rem !important;
    height: 2.3rem !important;
}}
[data-testid="stBottom"] [data-testid="stVerticalBlock"]:has(> .st-key-fill_random) {{
    position: relative !important;
}}
.st-key-fill_random {{
    position: absolute;
    inset-inline-end: 6.55rem;
    top: 50%;
    transform: translateY(-50%);
    z-index: 6;
    width: auto !important;
    margin: 0 !important;
}}
.st-key-fill_random [data-testid="stButton"] {{
    width: auto !important;
}}
.st-key-fill_random button {{
    width: 5.2rem !important;
    height: 5.2rem !important;
    min-width: 5.2rem !important;
    min-height: 5.2rem !important;
    padding: 0 !important;
    border-radius: 0.4rem !important;
    background: #1c83e1 !important;
    border-color: #1c83e1 !important;
    color: #fff !important;
    font-size: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}
.st-key-fill_random button:hover,
.st-key-fill_random button:focus {{
    background: #1566b0 !important;
    border-color: #1566b0 !important;
    color: #fff !important;
}}
.st-key-fill_random button > * {{
    display: none !important;
}}
.st-key-fill_random button::after {{
    content: "";
    display: block;
    flex: 0 0 3.825rem;
    width: 3.825rem !important;
    min-width: 3.825rem !important;
    height: 3.825rem !important;
    background: url("{dice_uri}") center / 118% 118% no-repeat;
}}
@keyframes trippy-ask-flash {{
    0%, 45% {{
        box-shadow: 0 0 0 2px rgba(255, 75, 75, 0.45);
    }}
    22%, 100% {{
        box-shadow: none;
    }}
}}
@keyframes trippy-ask-bold {{
    0%, 45% {{
        font-weight: 700;
        color: rgb(49, 51, 63);
    }}
    100% {{
        font-weight: 400;
    }}
}}
[data-testid="stChatInput"] > div {{
    animation: trippy-ask-flash 1.8s ease;
}}
[data-testid="stChatInputTextArea"]::placeholder {{
    animation: trippy-ask-bold 1.8s ease;
}}
@media (prefers-reduced-motion: reduce) {{
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInputTextArea"]::placeholder {{
        animation: none;
    }}
}}
</style>
"""


def apply_dir_script(lang: Lang) -> str:
    return f"""
<script>
const root = window.parent.document.documentElement;
root.setAttribute("dir", "ltr");
root.setAttribute("lang", "{lang}");
</script>
"""
