#!/bin/sh
# One-shot ingest (and migrate) inside the prod image.
# Usage: job.sh availability [--site 2]
set -eu
cd /app
export PYTHONPATH="/app:/app/source/scraper"
PY=/app/.venv/bin/python

job="${1:-}"
if [ -n "$job" ]; then
  shift
fi
if [ "${1:-}" = "--" ]; then
  shift
fi

case "$job" in
  availability)
    exec "$PY" source/scraper/populate_availability.py "$@"
    ;;
  reviews)
    exec "$PY" -m source.scraper.populate_reviews_and_claims "$@"
    ;;
  claims)
    exec "$PY" -m source.scraper.populate_claims "$@"
    ;;
  sites)
    exec "$PY" source/scraper/discover_sites.py "$@"
    ;;
  rooms)
    exec "$PY" -m source.scraper.rules_ingest.rooms "$@"
    ;;
  prices)
    exec "$PY" -m source.scraper.info_site.scrape --prices "$@"
    ;;
  rules)
    exec "$PY" -m source.scraper.rules_ingest.ingest "$@"
    ;;
  breadcrumbs)
    exec "$PY" -m source.scraper.info_site.breadcrumbs "$@"
    ;;
  place-ids)
    exec "$PY" source/scraper/populate_google_place_id.py "$@"
    ;;
  info)
    "$PY" -m source.scraper.rules_ingest.rooms "$@"
    "$PY" -m source.scraper.info_site.scrape --prices "$@"
    "$PY" -m source.scraper.rules_ingest.ingest "$@"
    exec "$PY" -m source.scraper.info_site.breadcrumbs "$@"
    ;;
  migrate)
    exec "$PY" -m alembic upgrade head
    ;;
  *)
    echo "usage: job.sh <availability|reviews|claims|sites|rooms|prices|rules|breadcrumbs|place-ids|info|migrate> [args]" >&2
    exit 2
    ;;
esac
