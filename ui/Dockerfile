# Streamlit boutique pipeline — talks to the FastAPI service in api/.
#
# The image carries Streamlit, plotly, the small src/ display helpers it
# imports (heatmap_at, classify_peak, …) and the PDF stack (reportlab +
# matplotlib + a Hebrew-capable TTF). It does NOT carry sample data —
# mount src/data/structuredData/ as a volume (see docker-compose.yml).
#
# Build from repo root:
#     docker build -f ui/Dockerfile -t elevator-ui .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# DejaVu ships Hebrew glyphs and is the last entry in the report's font
# fallback list, so it's enough to make the Hebrew PDF render correctly
# inside the container even when no system "real" Hebrew font exists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-ui.txt /app/requirements-ui.txt
RUN pip install -r /app/requirements-ui.txt

# The UI imports a few pure-numpy helpers from src/ (heatmap_at,
# classify_peak, …) plus the loaded-signal dataclass and the boutique
# Streamlit package. It does NOT call the segmentation / prediction
# algorithms directly — those go through the API.
COPY src/ /app/src/
COPY ui/ /app/ui/

# Step 1 ("How to use") of the wizard renders ~39 MB of PNG screenshots
# from docs/latex/figures/boutique/. They're static reference material so
# we bake them into the image rather than mounting a volume. ``_FIG_ROOT``
# in step1_howto.py resolves to ../../../docs/latex/figures/boutique
# relative to the Streamlit pipeline package, so the on-image path has to
# match.
COPY docs/latex/figures/boutique /app/docs/latex/figures/boutique

# Where the API lives. Override at run-time if the API is not on the
# default docker-compose service name (e.g. ``http://api.namespace:8000``
# inside Kubernetes, or ``http://host.docker.internal:8000`` for local
# debugging against an API on the host).
ENV API_URL=http://api:8000

EXPOSE 8501

# Streamlit binds to 0.0.0.0 so the host can reach it; usage stats
# disabled because there's nothing useful to send from a private pod.
CMD ["streamlit", "run", "src/pipelines/boutique_pipeline.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
