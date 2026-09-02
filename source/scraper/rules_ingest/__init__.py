"""Site-level rule and amenity ingestion from parks.org.il info pages."""

from source.scraper.rules_ingest.db import (
    ResolvedRule,
    sync_campsite_amenity_ids,
    upsert_campsite_rules,
)
from source.scraper.rules_ingest.fetch import fetch_page_html
from source.scraper.rules_ingest.llm import RuleExtractorLLMClient
from source.scraper.rules_ingest.schemas import RuleExtract, RuleStatement
from source.scraper.rules_ingest.sections import Section, parse_sections

__all__ = [
    "ResolvedRule",
    "RuleExtract",
    "RuleExtractorLLMClient",
    "RuleStatement",
    "Section",
    "fetch_page_html",
    "parse_sections",
    "sync_campsite_amenity_ids",
    "upsert_campsite_rules",
]
