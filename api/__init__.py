"""HTTP API exposing the segmentation and prediction stages.

The two endpoints are pure functions over JSON-serialisable accelerometer
data — no on-disk state, no session store. The Streamlit boutique UI in
``ui/`` is one client; downstream apps the dev team builds are others.
"""
