# Docker — Boutique UI

One container, one `docker-compose.yml`:

| Service | Image         | Port | What it is |
|---------|---------------|------|------------|
| `ui`    | `elevator-ui` | 8501 | Streamlit "boutique pipeline". Runs segmentation and prediction in-process (no separate API service) and does data input, plot rendering and PDF export. |

---

## Quickstart

```bash
# from the repo root
docker compose up --build
```

Open <http://localhost:8501>.

Stop with `Ctrl-C` then `docker compose down`.

---

## Ports — what's listening, where

| Container      | Internal port | What's served                               |
|----------------|----------------|----------------------------------------------|
| `elevator-ui`  | **8501**       | The Streamlit boutique pipeline (a web app)  |

That's the port you map to "the outside" on a pod / VM / load balancer.
In `docker-compose.yml` it's published as `8501:8501`, but the host-side
port is yours to pick.

### On a Kubernetes pod

The pod's `containerPort` is the **internal** port — the one the process
inside the container is listening on:

```yaml
spec:
  containers:
    - name: elevator-ui
      image: your-registry/elevator-ui:1.0
      ports:
        - name: http
          containerPort: 8501   # ← the one to map
---
apiVersion: v1
kind: Service
metadata:
  name: elevator-ui
spec:
  selector: { app: elevator-ui }
  ports:
    - port: 80          # what callers hit on the Service
      targetPort: 8501  # the containerPort above
```

For external exposure put an Ingress in front.

### What you get

Streamlit **is** a single-page web app — it speaks HTTP but isn't a REST
API, there's no `/segment` or `/predict` to call from another service:

```
GET /            — the Streamlit wizard (entry point)
GET /_stcore/*   — Streamlit's internal websocket + asset paths (don't call directly)
```

Open it in a browser; it's meant for humans walking through the 5-step
pipeline.

---

## "Phone DB" data input

`src/data/loadFromDB.py::loadDataFromS3` is currently a **stub** that reads a
fixed local experiment from `src/data/structuredData/data/<exp>/ACC.csv`. The
default `docker-compose.yml` mounts that folder read-only into the container
so the stub keeps working:

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
docker compose build ui
docker compose up -d
```

If the new backend doesn't need the local sample data anymore, drop the
volume mount in `docker-compose.yml` and the host folder can go away too.

If the new backend needs extra Python deps (e.g. `boto3`), add them to
`requirements-ui.txt` and rebuild.

---

## Updating code → image on a pod

The full path from a local code change to an updated pod running the new
image.

### What the Dockerfile copies in

Not just `src/`:

```
requirements-ui.txt    → pip install
src/                   → loadFromDB, the segmentation/prediction algorithms,
                          the streamlit pipeline package, display helpers
ui/                    → the in-process api_client wrapper module
```

The `.dockerignore` keeps junk out — `venv/`,
`src/data/structuredData/` (1.4 GB), `elevator_reports/`,
`__pycache__/`, etc. — so the build context stays small.

A code change matters only if it touches `src/`, `ui/`, or
`requirements-ui.txt`. Touching `docs/`, `scripts/`, `benchmarks/`, or
anything in `.dockerignore` doesn't affect the image.

### Step 1 — make the code change locally

Edit whatever file. No rebuild yet.

### Step 2 — rebuild

```bash
docker compose build ui
```

### Step 3 — smoke-test locally

```bash
docker compose up -d
# open http://localhost:8501 and click through the wizard
docker compose down
```

### Step 4 — tag for your registry

The local image is `elevator-ui:latest` (per `image:` in
`docker-compose.yml`). For a pod you need a registry-qualified name and a
real version — never ship `:latest`, pods cache by tag and `:latest` is a
debugging nightmare in production:

```bash
docker tag elevator-ui:latest your-registry/elevator-ui:1.2.0
# Replace "your-registry" with whatever you actually use:
#   ghcr.io/yourorg, gcr.io/yourproject, docker.io/yourdockerid, …
```

### Step 5 — push to the registry

```bash
docker push your-registry/elevator-ui:1.2.0
```

(First time on a new machine you may need `docker login your-registry`.)

### Step 6 — roll the pod onto the new image

Depends on how the pod is managed.

**Plain `kubectl` with a Deployment:**

```bash
kubectl set image deployment/elevator-ui \
  elevator-ui=your-registry/elevator-ui:1.2.0
kubectl rollout status deployment/elevator-ui
```

`kubectl` rolls pods one at a time — zero downtime if you have more than
one replica.

**Edit the YAML and re-apply:**

```bash
# bump the image tag in your-deployment.yaml, then:
kubectl apply -f your-deployment.yaml
```

**Helm:**

```bash
helm upgrade elevator-ui ./chart --set image.tag=1.2.0
```

### TL;DR copy/paste

```bash
# 1. edit code
# 2. rebuild + smoke-test
docker compose build ui
docker compose up -d
# open http://localhost:8501
docker compose down

# 3. tag + push
docker tag elevator-ui:latest your-registry/elevator-ui:1.2.0
docker push your-registry/elevator-ui:1.2.0

# 4. update the pod
kubectl set image deployment/elevator-ui \
  elevator-ui=your-registry/elevator-ui:1.2.0
kubectl rollout status deployment/elevator-ui
```

---

## Image-level details

### `ui/Dockerfile`

* Base: `python:3.11-slim`.
* Adds `fonts-dejavu` so the Hebrew PDF report renders with a TTF that has
  Hebrew glyphs (the report's font fallback list ends at DejaVu).
* Installs `requirements-ui.txt` (streamlit / plotly / matplotlib /
  reportlab / python-bidi / openpyxl + the numerics — numpy / pandas /
  scipy / pydantic).
* Entry: `streamlit run src/pipelines/boutique_pipeline.py …`.

### `.dockerignore`

`src/data/structuredData/`, `venv/`, `elevator_reports/`, etc. are excluded
from the build context. Without those excludes every `docker build` would
copy 1.4 GB of CSVs into the image.

---

## Common operations

| Goal | Command |
|------|---------|
| Build the image | `docker compose build` |
| Run in foreground (logs in terminal) | `docker compose up` |
| Run in background | `docker compose up -d` |
| Tail logs | `docker compose logs -f ui` |
| Restart after a code change | `docker compose up -d --build` |
| Stop and remove containers | `docker compose down` |
| Open Streamlit | <http://localhost:8501> |
