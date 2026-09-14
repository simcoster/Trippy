FROM python:3.14-slim

WORKDIR /app

# Install system dependencies for PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies and create virtual environment
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

RUN chmod +x /app/scripts/cloud/job.sh

EXPOSE 8501

CMD ["/app/.venv/bin/streamlit", "run", "scripts/streamlit_chat.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
