# Run from the repo root. Requires GNU make + uv.
# Pipeline: scrape-sites → scrape-booking-ids → scrape-prices → scrape-availability

export PYTHONPATH := source/scraper

.PHONY: help scrape-prices scrape-sites scrape-booking-ids scrape-availability scrape-all

help:
	@echo scrape-prices        info-site rate cards → accommodation_types + list_prices
	@echo scrape-sites         parks.org.il listing → campsites
	@echo scrape-booking-ids   INPA hotel ids → campsites.booking_hotel_id
	@echo scrape-availability  INPA vacancies → availability (match existing types)
	@echo scrape-all           sites, booking ids, prices, then availability

scrape-prices:
	uv run python source/scraper/info_site/scrape.py --prices

scrape-sites:
	uv run python source/scraper/discover_sites.py

scrape-booking-ids:
	uv run python source/scraper/populate_availability_id.py

scrape-availability:
	uv run python source/scraper/populate_availability.py

scrape-all: scrape-sites scrape-booking-ids scrape-prices scrape-availability
