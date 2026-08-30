import json
import os
import warnings
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Annotated, Any, Literal, TypedDict

# Suppress Pydantic V1 compatibility warning with Python 3.14+
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

import psycopg
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pgvector.psycopg import register_vector

from source.agent.constraints import (
    EMPTY_CONSTRAINTS,
    campsite_name_from_parsed,
    claim_recency,
    normalize_constraints,
    parse_constraints_dict,
    semantic_search_queries,
    today_il,
)
from source.scraper.amenity_enrichment.llm import (
    QWEN_INSTRUCT_30B_MODEL,
    QWEN_INSTRUCT_MODEL,
    ClaimsEmbeddingLLMClient,
    make_agent_chat_model,
)
from source.scraper.info_site.quote import quote_night
from source.scraper.info_site.schemas import RatePeriod

# Load environment variables
load_dotenv()


# ---- 1. State type for LangGraph ----

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---- 2. RAG Tool for Claims Search ----

_claims_embedder = ClaimsEmbeddingLLMClient()


def lookup_campsite_by_name(name: str) -> list[dict]:
    """Resolve a user-named park to campsite id(s). Not a catalog dump."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    terms = _campsite_lookup_terms(name)
    if not terms:
        return []
    like_patterns = [f"%{term}%" for term in terms]
    sql = """
        SELECT id, name, booking_hotel_id
        FROM campsites
        WHERE name ILIKE ANY(%s)
        ORDER BY id
        LIMIT 5
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (like_patterns,))
                rows = cur.fetchall()
        return [
            {
                "id": int(row[0]),
                "name": row[1],
                "hotel_id": int(row[0]),
                "booking_hotel_id": row[2],
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error looking up campsite: {e}"}]


OPEN_SLOTS_LIMIT = 80
AMENITY_MATCH_MAX_DISTANCE = -0.8
REJECTED_SAMPLE_LIMIT = 5
_LAST_OPEN_SLOTS_QUERY: dict[str, Any] | None = None


def _record_open_slots_query(record: dict[str, Any]) -> dict[str, Any]:
    global _LAST_OPEN_SLOTS_QUERY
    _LAST_OPEN_SLOTS_QUERY = record
    return record


def _stay_night_starts(date_range: dict) -> list[date] | None:
    """Check-in dates of each one-night row needed for [start, end)."""
    start = _parse_iso_date(date_range.get("start"))
    if start is None:
        return None
    end = _parse_iso_date(date_range.get("end"))
    if end is None or end <= start:
        return [start]
    nights: list[date] = []
    day = start
    while day < end:
        nights.append(day)
        day += timedelta(days=1)
    return nights


def _open_slots_sql(
    *,
    date_range: dict,
    site_id: int | list[int] | None,
    party_size: int | None,
    limit: int,
) -> tuple[str, list[Any]] | tuple[None, str]:
    nights = _stay_night_starts(date_range)
    if not nights:
        return None, "no_date"
    stay_start = nights[0]
    stay_end = nights[-1] + timedelta(days=1)
    clauses = [
        "a.start_date >= %s",
        "a.start_date < %s",
        "a.end_date = a.start_date + 1",
    ]
    params: list[Any] = [stay_start, stay_end]
    if isinstance(site_id, list):
        ids = [int(x) for x in site_id]
        if not ids:
            return None, "empty_site_ids"
        clauses.append("a.site_id = ANY(%s)")
        params.append(ids)
    elif site_id is not None:
        clauses.append("a.site_id = %s")
        params.append(int(site_id))
    if party_size is not None:
        clauses.append("(at.max_occupancy IS NULL OR at.max_occupancy >= %s)")
        params.append(int(party_size))
    params.append(len(nights))
    params.append(limit)
    sql = (
        "SELECT a.site_id, c.name, MIN(a.start_date), MAX(a.end_date),\n"
        "       MIN(a.room_count), at.id, at.name, at.max_occupancy\n"
        "FROM availability a\n"
        "JOIN accommodation_types at ON at.id = a.accommodation_type_id\n"
        "JOIN campsites c ON c.id = a.site_id\n"
        f"WHERE {' AND '.join(clauses)}\n"
        "GROUP BY a.site_id, c.name, at.id, at.name, at.max_occupancy\n"
        "HAVING COUNT(DISTINCT a.start_date) = %s\n"
        "ORDER BY MIN(a.start_date), at.id\n"
        "LIMIT %s"
    )
    return sql, params


def _sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "ARRAY[" + ", ".join(_sql_literal(v) for v in value) + "]"
    if hasattr(value, "isoformat"):
        value = value.isoformat()
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _render_sql(sql: str, params: list[Any]) -> str:
    parts = sql.split("%s")
    if len(parts) != len(params) + 1:
        return sql
    out = [parts[0]]
    for part, param in zip(parts[1:], params):
        out.append(_sql_literal(param))
        out.append(part)
    return "".join(out)


def _iso_day(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _parse_iso_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _rate_period_for_stay(date_range: dict | None) -> RatePeriod:
    if not isinstance(date_range, dict):
        return "weekday"
    start = _parse_iso_date(date_range.get("start"))
    if start is None:
        return "weekday"
    end = _parse_iso_date(date_range.get("end")) or (start + timedelta(days=1))
    day = start
    while day < end:
        if day.weekday() >= 5:
            return "weekend_holiday"
        day += timedelta(days=1)
    return "weekday"


def _price_per_night_constraint(
    numeric: list | None,
) -> tuple[str, float] | None:
    for item in numeric or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").lower()
        if field not in {"price_per_night", "price", "cost"}:
            continue
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            return None
        op = str(item.get("operator") or "=")
        return op, value
    return None


def _price_matches(price: float | None, constraint: tuple[str, float] | None) -> bool:
    if constraint is None:
        return True
    if price is None:
        return False
    op, bound = constraint
    if op in {"<=", "=<"}:
        return price <= bound
    if op in {">=", "=>"}:
        return price >= bound
    if op == "<":
        return price < bound
    if op == ">":
        return price > bound
    return price == bound


def _load_list_prices(type_ids: list[int]) -> dict[int, list[SimpleNamespace]]:
    if not type_ids:
        return {}
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return {}
    sql = """
        SELECT accommodation_type_id, guest_type, rate_period, price
        FROM list_prices
        WHERE accommodation_type_id = ANY(%s)
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (type_ids,))
                rows = cur.fetchall()
    except Exception:
        return {}
    by_type: dict[int, list[SimpleNamespace]] = {}
    for type_id, guest_type, rate_period, price in rows:
        by_type.setdefault(int(type_id), []).append(
            SimpleNamespace(
                guest_type=guest_type,
                rate_period=rate_period,
                price=float(price),
            )
        )
    return by_type


def _quote_slot_price(
    rates: list[SimpleNamespace],
    *,
    party_size: int | None,
    rate_period: RatePeriod,
) -> float | None:
    if not rates:
        return None
    adults = party_size if party_size and party_size > 0 else 1
    try:
        return float(quote_night(rates, adults=adults, rate_period=rate_period))
    except ValueError:
        return None


def search_open_slots(
    *,
    date_range: dict | None = None,
    site_id: int | list[int] | None = None,
    party_size: int | None = None,
    numeric_constraints: list | None = None,
    limit: int = OPEN_SLOTS_LIMIT,
) -> list[dict]:
    """Catalog vacancies for a stay window, with list-price quotes.

    Availability is stored as one-night rows. A multi-night stay matches
    only when the type has a row for every night in [start, end). Party
    size uses accommodation max_occupancy (scrape is 1-adult). Price
    filters use quote_night against list_prices. Optional site_id narrows
    to a named park.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        _record_open_slots_query({"skipped": "no_database_url"})
        return []
    if not isinstance(date_range, dict) or not date_range.get("start"):
        _record_open_slots_query({"skipped": "no_date"})
        return []
    built = _open_slots_sql(
        date_range=date_range,
        site_id=site_id,
        party_size=party_size,
        limit=limit,
    )
    if built[0] is None:
        _record_open_slots_query({"skipped": built[1]})
        return []
    sql, params = built
    query_record: dict[str, Any] = {
        "sql": _render_sql(sql, params),
        "price_constraint": _price_per_night_constraint(numeric_constraints),
        "rate_period": _rate_period_for_stay(date_range),
    }
    _record_open_slots_query(query_record)
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
    except Exception as e:
        query_record["error"] = str(e)
        return [{"error": f"Error searching availability: {e}"}]
    query_record["row_count"] = len(rows)

    slots: list[dict] = []
    for row in rows:
        occupancy = int(row[7]) if row[7] is not None else None
        slots.append(
            {
                "campsite_id": int(row[0]),
                "campsite": row[1],
                "start": _iso_day(row[2]),
                "end": _iso_day(row[3]),
                "room_count": int(row[4]),
                "accommodation_type_id": int(row[5]),
                "accommodation_type": row[6],
                "max_occupancy": occupancy,
                "occupancy_unknown": occupancy is None,
            }
        )
    type_ids = list({int(s["accommodation_type_id"]) for s in slots})
    prices = _load_list_prices(type_ids)
    rate_period = _rate_period_for_stay(date_range)
    price_constraint = _price_per_night_constraint(numeric_constraints)
    quoted: list[dict] = []
    for slot in slots:
        price = _quote_slot_price(
            prices.get(int(slot["accommodation_type_id"])) or [],
            party_size=party_size,
            rate_period=rate_period,
        )
        if not _price_matches(price, price_constraint):
            continue
        slot["price_per_night"] = price
        quoted.append(slot)
    query_record["quoted_count"] = len(quoted)
    return quoted


def search_availability(
    hotel_id: int,
    *,
    date_range: dict | None = None,
    party_size: int | None = None,
    limit: int = 50,
) -> list[dict]:
    """Vacancies for one campsite (campsites.id / accommodation_types.hotel_id)."""
    return search_open_slots(
        date_range=date_range,
        site_id=hotel_id,
        party_size=party_size,
        limit=limit,
    )


def _campsite_lookup_terms(name: str) -> list[str]:
    text = (name or "").strip()
    if not text:
        return []
    terms = [text]
    key = " ".join(text.lower().replace("-", " ").split())
    alias = _NAMED_CAMPSITE_ALIASES.get(key)
    if alias and alias not in terms:
        terms.append(alias)
    return terms


_NAMED_CAMPSITE_ALIASES = {
    "horshat tal": "חורשת טל",
    "horashat tal": "חורשת טל",
    "hurshat tal": "חורשת טל",
}


def _party_size_from_numeric(numeric: list) -> int | None:
    for item in numeric or []:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "").lower()
        if field not in {"party_size", "adults", "guests"}:
            continue
        try:
            return int(item.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def search_campsites(numeric_constraints):
    """
    List campsites from the 'campsites' table (id, name, url).
    Numeric filters (price / ride time) are not on this table yet;
    they will come from availability data later. `numeric_constraints`
    is accepted for API compatibility with the planner node.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return "Error: DATABASE_URL not configured"
    _ = numeric_constraints  # reserved for future availability filters
    sql = """
        SELECT id, name, url
        FROM campsites
        ORDER BY id
        LIMIT 50
    """
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                if not rows:
                    return "No campsites found"
                return [
                    {"id": row[0], "name": row[1], "url": row[2]}
                    for row in rows
                ]
    except Exception as e:
        return f"Error during search_campsites: {e}"


def _query_vec_literal(query: str) -> str:
    embedding = _claims_embedder.embed([query])[0]
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def search_stated_amenities(
    query: str,
    limit: int = 5,
    *,
    embedding: str | None = None,
    accommodation_type_ids: list[int] | None = None,
) -> list[dict]:
    """Rank accommodation types by closest official amenity embedding."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    if accommodation_type_ids is not None and not accommodation_type_ids:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    clauses = ["a.embedding IS NOT NULL", "at.amenities IS NOT NULL"]
    params: list[Any] = [vec_literal, vec_literal]
    if accommodation_type_ids is not None:
        clauses.append("at.id = ANY(%s)")
        params.append([int(x) for x in accommodation_type_ids])
    params.append(limit)
    sql = f"""
        SELECT at.id,
               at.name,
               at.hotel_id,
               MIN(a.embedding <#> %s::vector) AS distance,
               (array_agg(a.name ORDER BY a.embedding <#> %s::vector))[1]
                   AS matched_amenity
        FROM accommodation_types at
        CROSS JOIN LATERAL jsonb_array_elements(at.amenities) AS elem(val)
        JOIN amenities a ON a.id = (elem.val)::int
        WHERE {' AND '.join(clauses)}
        GROUP BY at.id, at.name, at.hotel_id
        ORDER BY distance
        LIMIT %s
    """
    try:
        with psycopg.connect(db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            {
                "amenity": row[4],
                "accommodation_type_id": int(row[0]),
                "accommodation_type": row[1],
                "hotel_id": int(row[2]),
                "distance": float(row[3]),
            }
            for row in rows
        ]
    except Exception as e:
        return [{"error": f"Error searching stated amenities: {e}"}]


def search_review_claims(
    query: str, limit: int = 5, *, embedding: str | None = None
) -> list[dict]:
    """Search review claims by vector similarity. Returns structured hits."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return []
    vec_literal = embedding or _query_vec_literal(query)
    sql = """
        SELECT campsite_id, claim_en, claim_he, review_date,
               embedding <#> %s::vector AS distance
        FROM claims
        WHERE claim_en IS NOT NULL OR claim_he IS NOT NULL
        ORDER BY embedding <#> %s::vector
        LIMIT %s
    """
    try:
        today = today_il()
        with psycopg.connect(db_url) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(sql, (vec_literal, vec_literal, limit))
                rows = cur.fetchall()
        hits: list[dict] = []
        for campsite_id, claim_en, claim_he, review_date, distance in rows:
            claim_text = claim_en or claim_he or "N/A"
            day, days_ago = claim_recency(review_date, today=today)
            hits.append(
                {
                    "claim": claim_text,
                    "campsite_id": campsite_id,
                    "date": day,
                    "days_ago": days_ago,
                    "distance": float(distance),
                }
            )
        return hits
    except Exception as e:
        return [{"error": f"Error searching claims: {e}"}]


def search_claims(query: str, limit: int = 5) -> str:
    """
    Search for review claims using vector similarity.

    Args:
        query: The search query (e.g., "fit for stargazing", "has hot water")
        limit: Maximum number of results to return (default: 5)

    Returns:
        A formatted string with matching claims, their campsite IDs, and relevance scores.
    """
    hits = search_review_claims(query, limit=limit)
    if not hits:
        return f"No claims found matching: {query}"
    if len(hits) == 1 and hits[0].get("error"):
        return str(hits[0]["error"])
    return "\n---\n".join(
        f"Campsite: {h.get('campsite_id')}\n"
        f"Claim: {h.get('claim')}\n"
        f"Date: {h.get('date')} ({h.get('days_ago')} days ago)\n"
        f"Relevance: {h.get('distance', 0):.4f}\n"
        for h in hits
    )


def _semantic_evidence_payload(queries: list[str], *, limit: int = 5) -> dict:
    """Official amenity names + dated review claims for the recommender."""
    if not queries:
        return {
            "query": "",
            "stated_amenities": [],
            "review_claims": [],
        }
    stated_amenities: list[str] = []
    review_claims: list[dict] = []
    seen_amenities: set[str] = set()
    seen_claims: set[tuple] = set()
    for query in queries:
        vec = _query_vec_literal(query)
        for hit in search_stated_amenities(query, limit=limit, embedding=vec):
            if hit.get("error"):
                continue
            label = str(hit.get("amenity") or "").strip()
            if label and label not in seen_amenities:
                seen_amenities.add(label)
                stated_amenities.append(label)
        for hit in search_review_claims(query, limit=limit, embedding=vec):
            if hit.get("error"):
                continue
            label = str(hit.get("claim") or "").strip()
            if not label:
                continue
            rec = {
                "claim": label,
                "date": hit.get("date"),
                "days_ago": hit.get("days_ago"),
                "campsite_id": hit.get("campsite_id"),
            }
            key = (rec["claim"], rec["date"], rec["campsite_id"])
            if key in seen_claims:
                continue
            seen_claims.add(key)
            review_claims.append(rec)
    return {
        "query": queries[0] if len(queries) == 1 else queries,
        "stated_amenities": stated_amenities,
        "review_claims": review_claims,
    }


# Create the tool
claims_search_tool = StructuredTool.from_function(
    func=search_claims,
    name="search_claims",
    description=(
        "Search for review claims about campsites using semantic similarity. "
        "Use this when users ask about specific features, amenities, or experiences "
        "at campsites (e.g., 'has hot water', 'good for stargazing', 'clean facilities') "
        "that are not numeric (like 'price < 100', 'rating > 4.5', 'distance < 100km', etc.). "
        "Returns matching claims with campsite IDs and relevance scores."
    ),
)


# ---- 3. Two models: light + heavy (Nebius Qwen instruct) ----

# Do NOT bind_tools on the chat models: extractor/recommender need text/JSON
# replies. Tools are invoked imperatively in planner_node. Binding tools made
# Qwen return empty content + tool_calls, which surfaced as blank agent replies.
light_model = make_agent_chat_model(temperature=0.7)
heavy_model = make_agent_chat_model(temperature=0.7)
# Query-constraint extract (the LLM that used to live in planner). Flip to
# QWEN_INSTRUCT_MODEL when we want 235B here too.
planner_model = make_agent_chat_model(
    temperature=0, model=QWEN_INSTRUCT_30B_MODEL
)
AGENT_CHAT_MODEL = QWEN_INSTRUCT_MODEL
PLANNER_CHAT_MODEL = QWEN_INSTRUCT_30B_MODEL

_EMPTY_REPLY_FALLBACK = (
    "לא הצלחתי להשלים תשובה כרגע. נסו לנסח שוב עם תאריך, מיקום או העדפה (למשל מים זורמים)."
)


def _message_text(content) -> str:
    """Normalize LC message content (str | list blocks | None) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(getattr(block, "text", None) or str(block))
        return "".join(parts)
    return str(content)


def _is_keep_decision(text: str) -> bool:
    """True if the cleaner decided to keep the message for the planner."""
    t = _message_text(text).strip().lower().strip("\"'`")
    if not t:
        return False
    first_line = t.splitlines()[0].strip()
    first_word = first_line.split()[0].strip(".,!?;:") if first_line else ""
    if first_word == "keep":
        return True
    if first_word == "drop":
        return False
    return t == "keep" or t.startswith("keep ")


def _parse_constraints_json(raw: str) -> dict:
    """Parse extractor JSON then normalize to date / numeric / semantic schema."""
    parsed = parse_constraints_dict(raw)
    if not parsed:
        return dict(EMPTY_CONSTRAINTS)
    return normalize_constraints(parsed)


def _constraints_from_tool_calls(tool_calls) -> dict:
    semantic: list[dict] = []
    for tc in tool_calls or []:
        name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
        args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
        if name == "search_claims" and isinstance(args, dict):
            query = args.get("query")
            if query:
                semantic.append({"query": query})
    return normalize_constraints(
        {
            "semantic_constraints": semantic,
            "numeric_constraints": [],
            "date": None,
        }
    )


def _latest_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            text = _message_text(msg.content).strip()
            if text:
                return text
    return ""


# ---- 4. Nodes ----

def router(state: ChatState) -> str:
    """
    Decide if the message is trivial (like "thanks!") or needs processing.
    """
    last_message = _message_text(state["messages"][-1].content).lower().strip()
    
    # Trivial responses that light model can handle directly
    trivial_patterns = [
        "thanks", "thank you", "thx", "ty", "appreciate it",
        "ok", "okay", "got it", "sounds good", "perfect",
        "bye", "goodbye", "see you", "later"
    ]
    
    if any(pattern in last_message for pattern in trivial_patterns) and len(last_message) < 50:
        return "trivial"
    else:
        return "non_trivial"


def light_node(state: ChatState) -> ChatState:
    """
    Light model handles:
    - Trivial prompts: answer directly
    - Non-trivial prompts: clean message to extract only trip-planning related content
    """
    last_message = state["messages"][-1]
    last_content = _message_text(last_message.content)
    
    # Check if this is a trivial response (from router)
    # We'll use a simple heuristic: if it's very short and matches trivial patterns
    trivial_patterns = [
        "thanks", "thank you", "thx", "ty", "appreciate it",
        "ok", "okay", "got it", "sounds good", "perfect",
        "bye", "goodbye", "see you", "later"
    ]
    is_trivial = any(pattern in last_content.lower() for pattern in trivial_patterns) and len(last_content) < 50
    
    if is_trivial:
        # Answer trivial prompts directly
        response = light_model.invoke(state["messages"])
        text = _message_text(response.content).strip() or _EMPTY_REPLY_FALLBACK
        return {"messages": [AIMessage(content=text)]}
    else:
        # Clean non-trip-planning content from the message
        # IMPORTANT: Consider the full conversation context - references to previous
        # messages (like "not redplace" after "redplace" was mentioned) are trip-planning related
        
        # Format conversation history for context
        conversation_context = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {_message_text(msg.content)}"
            for msg in state["messages"][:-1]
        ])
        
        cleaning_prompt = f"""
        You are a filter before a trip-planning assistant.
        You see the full conversation and the last user message (in Hebrew).

        Your job:
        - If the last user message is related to planning or updating a trip 
          (destinations, dates, people coming, budget, rides, packing, logistics,
          amenities such as running water, electricity, showers, etc.)
          OR it clearly refers to something trip-related mentioned earlier in the conversation
          (like "יהודה?" referring to "מדבר יהודה"),
          answer: "keep".
        - Otherwise (small talk, jokes, family, work, anything unrelated), answer: "drop".

        Always answer with exactly one word: "keep" or "drop".

        Examples:
        Conversation:
            User: "איפה יש מלון יפה?"
            Assistant: "יש מלון יפה במדבר יהודה"
            User: "יהודה?"
            Assistant: 
            → keep

        Conversation:
            User: "אני רוצה משהו לשישי הבא עם מים זורמים"
            Assistant:
            → keep

        Conversation:
            User: "מה שלומך?"
            Assistant: "בסדר, איך אני יכול לעזור?"
            User: "איפה אמא?"
            Assistant:
            → drop

        Conversation history:
        {conversation_context}

        Latest message: {last_content}
        """.strip()
        
        # Include full conversation history so cleaning can understand context
        cleaning_messages = state["messages"][:-1] + [HumanMessage(content=cleaning_prompt)]
        cleaned_response = light_model.invoke(cleaning_messages)
        cleaned_content = _message_text(cleaned_response.content)
        
        # On keep: add no messages so the user's HumanMessage stays last and
        # check_after_cleaning routes to the extractor (avoid injecting "keep").
        if _is_keep_decision(cleaned_content):
            return {}
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I didn't find any trip-planning related questions in your "
                        "message. How can I help you plan your trip?"
                    )
                )
            ]
        }


def check_after_cleaning(state: ChatState) -> str:
    """
    Check if there's still content after cleaning that needs heavy model.
    """
    last_message = state["messages"][-1]

    # If the last message is an AI response (from light model), we're done
    if isinstance(last_message, AIMessage):
        return "end"

    # If there's a cleaned human message, route to heavy model
    if isinstance(last_message, HumanMessage) and _message_text(last_message.content).strip():
        return "heavy"
    else:
        return "end"


def extractor_node(state: ChatState) -> ChatState:
    """Extract structured constraints from the user query (JSON only; no tools).

    Uses planner_model (30B). Flip that binding to 235B when we want it.
    """
    today = today_il()
    system_msg = SystemMessage(
        content=(
            f"""
            You are a structured query extractor for a campsite recommendation system called Trippy.
            Analyze the user query and extract constraints as JSON only (no commentary).

            Today's date (Asia/Jerusalem): {today.isoformat()}
            (weekday: {today.strftime("%A")})

            Schema (all keys required; use empty arrays / null when absent):
            {{
              "date": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} | null,
              "campsite": "Horashat Tal" | null,
              "numeric_constraints": [
                {{"field": "price_per_night", "operator": "<=", "value": 500}}
              ],
              "semantic_constraints": [
                {{"query": "hot showers"}},
                {{"query": "quiet"}},
                {{"op": "or", "values": ["near the sea", "near a body of water"]}}
              ]
            }}

            Rules:
            1. Output ONLY JSON.
            2. date: check-in/check-out as ISO start/end (end is exclusive).
               One night ⇒ end is the day after start. Always at least one night
               (never start == end). "this Friday" / "next Friday" → Friday check-in,
               Saturday check-out.
               Resolve relative phrases using today's date.
               Do NOT put dates in numeric_constraints or semantic_constraints.
            3. numeric_constraints: price, party size, distance (km), rating only — never dates.
            4. campsite: only when the user names a specific park to stay at
               (e.g. "2 rooms in Horshat Tal" → "Horashat Tal" / "חורשת טל").
               Do NOT put that name in semantic_constraints.
               Region/vibe ("near the sea", "Negev") stays in semantic_constraints;
               campsite stays null.
            5. semantic_constraints: features, amenities, location prefs, and vibes
               (hot showers, running water, near the sea, quiet, good for kids).
               Top-level list is AND. Use {{"op":"or","values":[...]}} for alternatives
               (e.g. "near the sea or some body of water").
               Each other item: {{"query": "..."}}.
               Prefer English labels: "hot showers", "running water", "near the sea".
               Do not emit an "amenities" key.
            6. Preserve negation in wording when stated.
            7. Do not invent constraints the user did not imply.

            Example:
            Input: "next friday, near the sea or some body of water to swim in"
            Output:
            {{
              "date": {{"start": "<that Friday ISO>", "end": "<Saturday after that Friday>"}},
              "campsite": null,
              "numeric_constraints": [],
              "semantic_constraints": [
                {{"op": "or", "values": ["near the sea", "near a body of water"]}}
              ]
            }}
            """.strip().replace("            ", "")
        )
    )

    response = planner_model.invoke([system_msg] + state["messages"])
    raw = _message_text(response.content)
    tool_calls = getattr(response, "tool_calls", None) or []
    user_text = _latest_user_text(state["messages"])

    constraints_json = normalize_constraints(
        parse_constraints_dict(raw),
        today=today,
        user_text=user_text,
    )
    constraints_json = _attach_campsite(constraints_json, parse_constraints_dict(raw))
    # If the model emitted tool calls instead of JSON, recover queries from args.
    if (
        not constraints_json.get("semantic_constraints")
        and not constraints_json.get("numeric_constraints")
        and constraints_json.get("date") is None
        and tool_calls
    ):
        constraints_json = _constraints_from_tool_calls(tool_calls)
        constraints_json = normalize_constraints(
            constraints_json, today=today, user_text=user_text
        )
        constraints_json = _attach_campsite(
            constraints_json, parse_constraints_dict(raw)
        )

    constraints_payload = json.dumps(constraints_json, ensure_ascii=False)
    if not constraints_payload.strip():
        constraints_payload = json.dumps(EMPTY_CONSTRAINTS, ensure_ascii=False)
    return {"messages": [AIMessage(content=constraints_payload)]}


def _attach_campsite(constraints: dict, parsed: dict) -> dict:
    name = campsite_name_from_parsed(parsed)
    if name:
        constraints["campsite"] = name
    return constraints


def _latest_constraints_json(messages: list[BaseMessage]) -> dict:
    """Read the most recent constraints AIMessage (from extractor_node)."""
    import json

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        raw = _message_text(msg.content).strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = parse_constraints_dict(raw)
        if isinstance(data, dict) and (
            "semantic_constraints" in data
            or "numeric_constraints" in data
            or "date" in data
            or "amenities" in data
            or "campsite" in data
        ):
            return _attach_campsite(normalize_constraints(data), data)
    return dict(EMPTY_CONSTRAINTS)


def _amenity_hit_matches(hit: dict) -> bool:
    if hit.get("error") or hit.get("accommodation_type_id") is None:
        return False
    dist = hit.get("distance")
    if dist is None:
        return True
    return float(dist) <= AMENITY_MATCH_MAX_DISTANCE


def _semantic_why_by_type(
    type_ids: list[int],
    semantic_constraints: list,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list[dict[str, Any]]]]:
    """AND-groups of amenity queries → (matched why, rejected why) by type id."""
    unique_ids = list(dict.fromkeys(int(x) for x in type_ids))
    if not unique_ids:
        return {}, {}
    groups = semantic_search_queries(semantic_constraints)
    if not groups:
        return {tid: [] for tid in unique_ids}, {}

    why_by_type: dict[int, list[dict[str, Any]]] = {tid: [] for tid in unique_ids}
    missing_by_type: dict[int, list[dict[str, Any]]] = {tid: [] for tid in unique_ids}
    matching = set(unique_ids)
    for queries in groups:
        group_hits: dict[int, dict[str, Any]] = {}
        for query in queries:
            vec = _query_vec_literal(query)
            hits = search_stated_amenities(
                query,
                limit=max(len(unique_ids), 1),
                embedding=vec,
                accommodation_type_ids=unique_ids,
            )
            for hit in hits:
                if not _amenity_hit_matches(hit):
                    continue
                tid = int(hit["accommodation_type_id"])
                if tid not in group_hits:
                    group_hits[tid] = {
                        "query": query,
                        "stated_amenity": hit.get("amenity"),
                        "distance": hit.get("distance"),
                    }
        matching &= set(group_hits)
        label = queries[0] if len(queries) == 1 else list(queries)
        for tid in unique_ids:
            if tid in group_hits:
                why_by_type[tid].append(group_hits[tid])
            else:
                missing_by_type[tid].append(
                    {
                        "reason": "missing_stated_amenity",
                        "query": label,
                    }
                )
    return (
        {tid: why_by_type[tid] for tid in matching},
        {tid: missing_by_type[tid] for tid in unique_ids if tid not in matching},
    )


def _review_claims_for_sites(
    site_ids: set[str],
    semantic_constraints: list,
    *,
    per_query_limit: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    if not site_ids:
        return {}
    by_site: dict[str, list[dict[str, Any]]] = {sid: [] for sid in site_ids}
    seen: set[tuple] = set()
    for queries in semantic_search_queries(semantic_constraints):
        for query in queries:
            vec = _query_vec_literal(query)
            for hit in search_review_claims(
                query, limit=per_query_limit, embedding=vec
            ):
                if hit.get("error"):
                    continue
                cid = str(hit.get("campsite_id") or "")
                if cid not in by_site:
                    continue
                rec = {
                    "query": query,
                    "claim": hit.get("claim"),
                    "date": hit.get("date"),
                    "days_ago": hit.get("days_ago"),
                }
                key = (cid, rec["claim"], rec["date"])
                if key in seen:
                    continue
                seen.add(key)
                by_site[cid].append(rec)
    return by_site


def _named_site_ids(name: str) -> tuple[list[int], dict[str, Any] | None]:
    hits = lookup_campsite_by_name(name)
    if not hits:
        return [], {"error": "No campsite matched that name", "query": name}
    if hits[0].get("error"):
        return [], {"error": hits[0]["error"], "query": name}
    ids = [
        int(hit["hotel_id"])
        for hit in hits
        if hit.get("hotel_id") is not None
    ]
    if not ids:
        return [], {"error": "No campsite matched that name", "query": name}
    return ids, None


def _planner_fits_payload(constraints_json: dict) -> dict[str, Any]:
    date_range = constraints_json.get("date")
    numeric = constraints_json.get("numeric_constraints") or []
    semantic = constraints_json.get("semantic_constraints") or []
    payload: dict[str, Any] = {
        "fits": [],
        "rejected": [],
        "rejected_count": 0,
        "constraints": constraints_json,
    }
    if not (isinstance(date_range, dict) and date_range.get("start")):
        payload["skipped"] = "no_date"
        return payload

    site_id: int | list[int] | None = None
    named = constraints_json.get("campsite")
    if named:
        site_ids, error = _named_site_ids(str(named))
        if error:
            payload["error"] = error["error"]
            payload["query"] = error.get("query")
            return payload
        site_id = site_ids if len(site_ids) > 1 else site_ids[0]

    slots = search_open_slots(
        date_range=date_range,
        site_id=site_id,
        party_size=_party_size_from_numeric(numeric),
        numeric_constraints=numeric,
    )
    if _LAST_OPEN_SLOTS_QUERY is not None:
        payload["open_slots_query"] = _LAST_OPEN_SLOTS_QUERY
    if slots and slots[0].get("error"):
        payload["error"] = slots[0]["error"]
        return payload

    type_ids = [int(s["accommodation_type_id"]) for s in slots]
    why_by_type, reject_why_by_type = _semantic_why_by_type(type_ids, semantic)
    fits: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def _slot_row(slot: dict, why: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "campsite_id": slot["campsite_id"],
            "campsite": slot["campsite"],
            "accommodation_type_id": int(slot["accommodation_type_id"]),
            "accommodation_type": slot["accommodation_type"],
            "start": slot["start"],
            "end": slot["end"],
            "room_count": slot.get("room_count"),
            "max_occupancy": slot.get("max_occupancy"),
            "occupancy_unknown": slot.get("occupancy_unknown"),
            "price_per_night": slot.get("price_per_night"),
            "why": why,
        }

    for slot in slots:
        tid = int(slot["accommodation_type_id"])
        if tid in why_by_type:
            fits.append(_slot_row(slot, why_by_type[tid]))
        else:
            rejected.append(
                _slot_row(
                    slot,
                    reject_why_by_type.get(tid)
                    or [{"reason": "semantic_mismatch"}],
                )
            )

    if fits and semantic:
        site_keys = {str(f["campsite_id"]) for f in fits}
        claims = _review_claims_for_sites(site_keys, semantic)
        for fit in fits:
            extra = claims.get(str(fit["campsite_id"])) or []
            if extra:
                fit["review_claims"] = extra

    payload["fits"] = fits
    payload["rejected"] = rejected[:REJECTED_SAMPLE_LIMIT]
    payload["rejected_count"] = len(rejected)
    return payload


def planner_node(state: ChatState) -> ChatState:
    """Vacancies + prices, then semantic intersection with evidence."""
    constraints_json = _latest_constraints_json(state["messages"])
    payload = _planner_fits_payload(constraints_json)
    return {
        "messages": [
            ChatMessage(
                content=json.dumps(payload, ensure_ascii=False, default=str),
                role="assistant",
            )
        ]
    }


def recommender_node(state: ChatState) -> ChatState:
    """Generate final recommendation based on constraints and tool responses."""
    
    system_msg = SystemMessage(
        content=(
            "You are a helpful trip-planning assistant for Trippy. "
            "Recommend only from the planner JSON field `fits`. "
            "Each fit is an available stay that already matches dates, party size, "
            "and price when those were given. "
            "Use `why` as official-listing evidence for requested features "
            "(stated amenity names). "
            "`review_claims` are guest reviews — lived quality, with date and "
            "days_ago. Weigh recent reviews more. When official amenities and "
            "reviews conflict, still consider the site but mention the caveat. "
            "Do not invent campsites, prices, or amenities that are not in `fits`. "
            "`rejected` is a short sample of open stays that failed a feature "
            "check; use `why` there only to explain misses, never to recommend. "
            "If `fits` is empty, say so clearly and ask a short follow-up "
            "(dates, area, budget, amenities). "
            "Never reply with an empty message. "
            "Respond in the same language as the user's query."
        )
    )
    
    # Get the original user messages and tool results from state
    # The state contains: user messages + constraint extraction message + tool messages
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ChatMessage)]
    
    # Generate recommendation using original user query + tool results
    recommendation = heavy_model.invoke([system_msg] + user_messages + tool_messages)
    text = _message_text(recommendation.content).strip() or _EMPTY_REPLY_FALLBACK

    return {"messages": [AIMessage(content=text)]}


# ---- 5. Build the graph ----

HeavyThrough = Literal["extractor", "planner", "recommender"]


def build_graph(*, stop_after: HeavyThrough = "recommender"):
    """Compile the agent. `stop_after` is the last heavy node to run (Streamlit)."""
    builder = StateGraph(ChatState)

    builder.add_node("light", light_node)
    builder.add_node("extractor", extractor_node)
    if stop_after in ("planner", "recommender"):
        builder.add_node("planner", planner_node)
    if stop_after == "recommender":
        builder.add_node("recommender", recommender_node)

    builder.add_conditional_edges(
        START,
        router,
        {
            "trivial": "light",
            "non_trivial": "light",
        },
    )
    builder.add_conditional_edges(
        "light",
        check_after_cleaning,
        {
            "heavy": "extractor",
            "end": END,
        },
    )

    if stop_after == "extractor":
        builder.add_edge("extractor", END)
    elif stop_after == "planner":
        builder.add_edge("extractor", "planner")
        builder.add_edge("planner", END)
    else:
        builder.add_edge("extractor", "planner")
        builder.add_edge("planner", "recommender")
        builder.add_edge("recommender", END)

    return builder.compile()


# Production / Telegram: full path
graph = build_graph()
