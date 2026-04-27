# Docker stack — API + Boutique UI

Two containers, one `docker-compose.yml`:

| Service | Image            | Port | What it is |
|---------|------------------|------|------------|
| `api`   | `elevator-api`   | 8000 | FastAPI service exposing the **segmentation** and **prediction** endpoints. Pure logic, stateless — drop this on a pod and build your app around it. |
| `ui`    | `elevator-ui`    | 8501 | Streamlit "boutique pipeline" — reference client. Calls the API for all logic; only does data input, plot rendering and PDF export locally. |

The two services are independent. The dev team only needs the **`api`** image
in production; the UI is optional reference material.

---

## Quickstart

```bash
# from the repo root
docker compose up --build
```

Once the API health check passes (5–15 s), open:

* **UI**:  <http://localhost:8501>
* **API docs (Swagger)**:  <http://localhost:8000/docs>
* **API health**:  <http://localhost:8000/health>

Stop with `Ctrl-C` then `docker compose down`.

---

## Ports — what's listening, where

Both containers expose **one HTTP port each** inside the container:

| Container       | Internal port | What's served                                 |
|-----------------|---------------|-----------------------------------------------|
| `elevator-api`  | **8000**      | The FastAPI app — the JSON endpoints          |
| `elevator-ui`   | **8501**      | The Streamlit boutique pipeline (a web app)   |

Those are the ports you map to "the outside" on a pod / VM / load balancer.
In `docker-compose.yml` they're published as `8000:8000` and `8501:8501`,
but the host-side port is yours to pick.

### On a Kubernetes pod

The pod's `containerPort` is the **internal** port — the one the process
inside the container is listening on. So for the API pod:

```yaml
spec:
  containers:
    - name: elevator-api
      image: your-registry/elevator-api:1.0
      ports:
        - name: http
          containerPort: 8000   # ← the one to map
      livenessProbe:
        httpGet:
          path: /health
          port: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: elevator-api
spec:
  selector: { app: elevator-api }
  ports:
    - port: 80          # what callers hit on the Service
      targetPort: 8000  # the containerPort above
```

Internal callers then reach the API at `http://elevator-api:80` (or `:8000`
if you set `port: 8000`); for external exposure put an Ingress in front.

The UI follows the same pattern with `containerPort: 8501`.

### The endpoints you actually call

**`elevator-api` (port 8000)** — REST endpoints your dev team's app will hit:

```
GET  /health         — liveness probe ({"status":"ok"})
POST /segment        — accelerometer → ride intervals + diagnostic state
POST /predict        — accelerometer + ride intervals → Δh per algorithm
GET  /docs           — Swagger UI (interactive API explorer)
GET  /openapi.json   — machine-readable schema for code generators
GET  /redoc          — alternate API docs
```

**`elevator-ui` (port 8501)** — Streamlit **is** a single-page web app. It
speaks HTTP, but it's not a REST API — no `/segment` or `/predict` here.
The whole UI is one URL:

```
GET /            — the Streamlit wizard (entry point)
GET /_stcore/*   — Streamlit's internal websocket + asset paths (don't call directly)
```

Open it in a browser. There's no "endpoint" you POST to from another
service — that's what the API is for. The UI is meant for humans walking
through the 5-step pipeline; under the hood it calls `elevator-api:8000`.

### How to actually run them

**Local (laptop)** — what you already have:

```bash
docker compose up --build
# UI:    http://localhost:8501
# API:   http://localhost:8000/docs
```

**API only on a pod**, dev team builds the app — ship just the API image:

```bash
docker build -f api/Dockerfile -t your-registry/elevator-api:1.0 .
docker push your-registry/elevator-api:1.0
# Deploy with containerPort: 8000 (see YAML above)
```

**Both on a pod** (e.g. internal demo / labelling tool) — ship both images.
The UI container needs `API_URL` set to wherever the API is reachable
inside the cluster — by default it points at `http://api:8000` (the
docker-compose service name), so for K8s set:

```yaml
env:
  - name: API_URL
    value: http://elevator-api:80    # the Service name + port from above
```

Otherwise the UI will try to reach `http://api:8000` and fail — there's no
service called `api` in your cluster.

### TL;DR

* **API container port: `8000`.** Map this on the pod.
* **UI container port: `8501`.** Map this if you want humans to use the
  Streamlit wizard.
* **The UI has no REST endpoints** — it's a web UI you load in a browser.
  The API is what your dev team's app talks to.
* **The UI must know where the API is**, via the `API_URL` env var.
  Default `http://api:8000` works inside `docker-compose`; in K8s set it
  to your Service DNS (e.g.
  `http://elevator-api.default.svc.cluster.local:80`).

---

## What the API exposes

### `POST /segment`

Detect ride intervals in an accelerometer trace.

```bash
curl -X POST http://localhost:8000/segment \
  -H "Content-Type: application/json" \
  -d '{
    "acc": {
      "timestamp_ms": [0, 20, 40, ...],
      "x": [0.01, 0.02, ...],
      "y": [0.0, 0.0, ...],
      "z": [9.81, 9.80, ...]
    },
    "phone_model": "",
    "include_state": true
  }'
```

Response shape:

```json
{
  "predictions": [
    { "t_start_s": 478.99, "t_end_s": 482.45,
      "ride_type": "down", "duration_s": 3.46,
      "joint_r2_mean": 0.962,
      "lobe1": { "...": "..." }, "lobe2": { "...": "..." } }
  ],
  "t0_ms": 1776248443775.0,
  "state": {
    "t":         [...], "a_vert":      [...], "a_smooth":    [...],
    "best_pos_r2":[...],"best_neg_r2": [...],
    "grid_w_s":  [...], "grid_f":      [...],
    "config":    { "r2_peak_thresh": 0.40, "...": "..." },
    "...":       "..."
  }
}
```

* `state` is the full diagnostic bundle — set `"include_state": false` to skip
  it (~1–2 MB on a 5-minute trace, mostly R² arrays).
* `t0_ms` is the epoch the time-axis is rebased to. The UI doesn't actually
  need to send it back to `/predict` — the predictor re-derives it from the
  ACC stream.
* `phone_model` is optional. When set, the detector tightens its amplitude
  floors to the phone's chip noise σ.

### `POST /predict`

Run **both** accelerometer-only Δh estimators (Trapezoid pulse-pair, ZUPT) on
a list of ride intervals.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "acc":      { "timestamp_ms": [...], "x": [...], "y": [...], "z": [...] },
    "segments": [ { "type": "up",   "start_s": 12.4, "end_s": 18.7 },
                  { "type": "down", "start_s": 25.1, "end_s": 33.9 } ],
    "phone_model": ""
  }'
```

Response:

```json
{
  "rows_by_algo": {
    "trap": [
      { "segment": 0, "type": "up",   "start_s": 12.4, "end_s": 18.7,
        "duration_s": 6.3, "delta_height_m": 4.21, "abs_height_m": 4.21,
        "ci_half_width": 0.55, "quality_score": 87.2,
        "accepted": true, "reject_reason": "" }
    ],
    "zupt": [ { "...": "..." } ]
  },
  "primary": "trap"
}
```

* The endpoint slices the segment ACC and a ±5 s stationary pre/post window
  internally — callers only have to send the full session ACC and the
  `(type, start_s, end_s)` triples.
* Restrict the algorithm set with `"algorithms": ["trap"]` if you only want
  one. Default is both.

A full live OpenAPI / Swagger reference is at <http://localhost:8000/docs>
once the stack is up.

---

## Accessing the Streamlit UI

* Open <http://localhost:8501>.
* The wizard is unchanged — How-to → Data → Segmentation → Prediction →
  Report. Steps 3 / 4 now call the API in `api/` instead of running the
  algorithms in-process; step 5 (PDF report) runs locally because it's just
  data layout.
* The UI talks to the API at `http://api:8000` from inside the docker
  network. Override with `API_URL=http://...` when running the UI container
  somewhere else.

### "Phone DB" data input

`src/data/loadFromDB.py::loadDataFromS3` is currently a **stub** that reads a
fixed local experiment from `src/data/structuredData/data/<exp>/ACC.csv`. The
default `docker-compose.yml` mounts that folder read-only into the UI
container so the stub keeps working:

```yaml
ui:
  volumes:
    - ./src/data/structuredData:/app/src/data/structuredData:ro
```

The dataset is ~1.4 GB and is **not** baked into the image — keep it on the
host (or replace the function entirely; see below).

---

## Replacing `loadDataFromS3`

When the real S3 backend is ready you swap the body of
`src/data/loadFromDB.py::loadDataFromS3`. Keep the signature and the return
type (`LoadedSignal` with an `acc` DataFrame in
`timestamp_ms,x,y,z` schema) so the rest of the UI keeps working unchanged.

After editing the function:

```bash
# Rebuild only the UI image (the API doesn't import loadDataFromS3).
docker compose build ui

# Bring the stack back up.
docker compose up -d
```

If the new backend doesn't need the local sample data anymore, drop the
volume mount in `docker-compose.yml` and the host folder can go away too.

If the new backend needs extra Python deps (e.g. `boto3`), add them to
`requirements-ui.txt` and rebuild — the UI image is the one that imports
`loadDataFromS3`.

---

## Updating code → image on a pod

The full path from a local code change to an updated pod running the new
image. Pick the matching rebuild target depending on what you touched.

### What each Dockerfile copies in

Not just `src/`. Each image has a slightly different scope:

**`api/Dockerfile`** copies:

```
requirements-api.txt   → pip install
src/                   → algorithm code (segmentation, prediction, utils, data)
api/                   → the FastAPI app itself (main.py, schemas.py, encoding.py)
```

**`ui/Dockerfile`** copies:

```
requirements-ui.txt    → pip install
src/                   → loadFromDB, the streamlit pipeline package, display helpers
ui/                    → the API client (api_client.py)
```

The `.dockerignore` keeps junk out — `venv/`,
`src/data/structuredData/` (1.4 GB), `elevator_reports/`,
`__pycache__/`, etc. — so the build context stays small.

A code change matters only if it touches `src/`, `api/`, `ui/`, or the
two `requirements-*.txt` files. Touching `docs/`, `scripts/`,
`benchmarks/`, or anything in `.dockerignore` doesn't affect either image.

### Step 1 — make the code change locally

Edit whatever file. No rebuild yet.

### Step 2 — rebuild the affected image(s)

| What you changed                              | Rebuild              |
|-----------------------------------------------|----------------------|
| `api/`, `src/segmentation/`, `src/prediction/`, `src/utils/` | `docker compose build api` |
| `ui/`, `src/pipelines/streamlit/`             | `docker compose build ui`  |
| `src/data/loadFromDB.py`                      | `docker compose build ui`  |
| `src/utils/`, `src/data/` (touches both)      | `docker compose build`     |
| `requirements-api.txt`                        | `docker compose build api` |
| `requirements-ui.txt`                         | `docker compose build ui`  |

### Step 3 — smoke-test locally

Bring the stack up and verify nothing's broken before shipping:

```bash
docker compose up -d
curl http://localhost:8000/health        # → {"status":"ok"}
# open http://localhost:8501 and click through the wizard
docker compose down
```

### Step 4 — tag for your registry

The local image is `elevator-api:latest` (per `image:` in
`docker-compose.yml`). For a pod you need a registry-qualified name and a
real version — never ship `:latest`, pods cache by tag and `:latest` is a
debugging nightmare in production:

```bash
docker tag elevator-api:latest your-registry/elevator-api:1.2.0
# Replace "your-registry" with whatever you actually use:
#   ghcr.io/yourorg, gcr.io/yourproject, docker.io/yourdockerid, …
```

### Step 5 — push to the registry

```bash
docker push your-registry/elevator-api:1.2.0
```

(First time on a new machine you may need `docker login your-registry`.)

### Step 6 — roll the pod onto the new image

Depends on how the pod is managed.

**Plain `kubectl` with a Deployment:**

```bash
kubectl set image deployment/elevator-api \
  elevator-api=your-registry/elevator-api:1.2.0
kubectl rollout status deployment/elevator-api
```

`kubectl` rolls pods one at a time and waits for the liveness probe on
`/health` to pass before killing the old one — zero downtime.

**Edit the YAML and re-apply:**

```bash
# bump the image tag in your-deployment.yaml, then:
kubectl apply -f your-deployment.yaml
```

**Helm:**

```bash
helm upgrade elevator-api ./chart --set image.tag=1.2.0
```

### TL;DR copy/paste

```bash
# 1. edit code
# 2. rebuild + smoke-test
docker compose build api          # or "ui", or just "build" for both
docker compose up -d
curl http://localhost:8000/health
docker compose down

# 3. tag + push
docker tag elevator-api:latest your-registry/elevator-api:1.2.0
docker push your-registry/elevator-api:1.2.0

# 4. update the pod
kubectl set image deployment/elevator-api \
  elevator-api=your-registry/elevator-api:1.2.0
kubectl rollout status deployment/elevator-api
```

Same flow for the UI image — swap `api` for `ui` and
`elevator-api` for `elevator-ui`.

---

## Image-level details

### `api/Dockerfile`

* Base: `python:3.11-slim`.
* Installs `requirements-api.txt` (numpy / pandas / scipy / pydantic /
  fastapi / uvicorn) and copies `src/` + `api/`.
* Entry: `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

### `ui/Dockerfile`

* Base: `python:3.11-slim`.
* Adds `fonts-dejavu` so the Hebrew PDF report renders with a TTF that has
  Hebrew glyphs (the report's font fallback list ends at DejaVu).
* Installs `requirements-ui.txt` (streamlit / plotly / requests / matplotlib
  / reportlab / python-bidi / openpyxl + the numerics).
* Entry: `streamlit run src/pipelines/boutique_pipeline.py …`.

### `.dockerignore`

`src/data/structuredData/`, `venv/`, `elevator_reports/`, etc. are excluded
from the build context. Without those excludes every `docker build` would
copy 1.4 GB of CSVs into the image.

---

## Common operations

| Goal | Command |
|------|---------|
| Build both images | `docker compose build` |
| Build a single image | `docker compose build api` / `docker compose build ui` |
| Run in foreground (logs in terminal) | `docker compose up` |
| Run in background | `docker compose up -d` |
| Tail logs of one service | `docker compose logs -f api` |
| Restart after a code change | `docker compose up -d --build` |
| Stop and remove containers | `docker compose down` |
| Hit the API directly | `curl http://localhost:8000/health` |
| Open Streamlit | <http://localhost:8501> |
| Run the API alone (no UI) | `docker compose up api` |
| Pin to a different API host | `API_URL=http://1.2.3.4:8000 docker compose up ui` |

---

## Deploying just the API

The API image has no dependency on Streamlit, plotly, reportlab, fonts or
the local dataset. It's safe to ship on its own:

```bash
docker build -f api/Dockerfile -t my-registry/elevator-api:1.0 .
docker push my-registry/elevator-api:1.0
```

Then point your app at it. A minimal Kubernetes Deployment needs only:

* The image above
* `containerPort: 8000`
* A liveness probe on `GET /health` (returns `{"status":"ok"}`)

No volumes, no secrets, no sidecars.
