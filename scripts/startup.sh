#!/bin/sh

echo "Starting ngrok..."
ngrok http 8000 &

echo "Starting FastAPI app..."
exec /app/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
