"""Boutique Streamlit UI — talks to the FastAPI service in ``api/``.

The UI used to call :func:`predict_intervals` and :class:`Predictor`
directly. With the API split it only calls the HTTP endpoints; this
package holds the small client wrapper they share.
"""
