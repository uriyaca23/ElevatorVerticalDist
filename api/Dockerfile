# FastAPI service exposing /segment and /predict.
#
# The image only carries the algorithm code under src/ — no Streamlit, no
# matplotlib, no fonts. Mount nothing; the API is stateless. Build context
# is the repo root (so requirements-api.txt and src/ are in scope):
#
#     docker build -f api/Dockerfile -t elevator-api .
FROM python:3.11-slim

# Faster, predictable Python + smaller pip footprint.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install requirements first so pip's layer is cached across src/ edits.
COPY requirements-api.txt /app/requirements-api.txt
RUN pip install -r /app/requirements-api.txt

# Algorithm code + the API package itself.
COPY src/ /app/src/
COPY api/ /app/api/

EXPOSE 8000

# uvicorn binds to all interfaces so docker-compose / Kubernetes can route
# to the container. Single worker is fine for the boutique-pipeline use
# case; scale horizontally with replicas if you need more throughput.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
