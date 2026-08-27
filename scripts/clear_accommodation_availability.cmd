@echo off
cd /d "%~dp0\.."
uv run python scripts\clear_accommodation_availability.py %*
