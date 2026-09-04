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

# Copy startup script
COPY scripts/startup.sh /start.sh
RUN chmod +x /start.sh

# Expose port
EXPOSE 8000

# Start the app
CMD ["sh", "/start.sh"]

