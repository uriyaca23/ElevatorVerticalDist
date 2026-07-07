# Streamlit boutique pipeline — runs segmentation/prediction in-process.
#
# The image carries Streamlit, plotly, the boutique Streamlit package and the
# `pyramidElevatorDist` package it sources ALL segmentation/prediction/signal
# logic from, plus the PDF stack (reportlab + matplotlib + a Hebrew-capable
# TTF). It does NOT carry sample data — mount src/data/structuredData/ as a
# volume (see docker-compose.yml).
#
# Build from repo root:
#     docker build -t elevator-ui .
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

# The UI runs the algorithms in-process (no separate API service) but only
# ever calls the `pyramidElevatorDist` public package, which in turn imports
# the segmentation/prediction code under src/. Both must be on the image.
COPY src/ /app/src/
COPY pyramidElevatorDist/ /app/pyramidElevatorDist/

# Step 1 ("How to use") of the wizard renders ~39 MB of PNG screenshots
# from docs/latex/figures/boutique/. They're static reference material so
# we bake them into the image rather than mounting a volume. ``_FIG_ROOT``
# in step1_howto.py resolves to ../../../docs/latex/figures/boutique
# relative to the Streamlit pipeline package, so the on-image path has to
# match.
COPY docs/latex/figures/boutique /app/docs/latex/figures/boutique

EXPOSE 8501

# Streamlit binds to 0.0.0.0 so the host can reach it; usage stats
# disabled because there's nothing useful to send from a private pod.
CMD ["streamlit", "run", "src/pipelines/boutique_pipeline.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--browser.gatherUsageStats=false"]
