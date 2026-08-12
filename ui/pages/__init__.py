"""Per-page Streamlit render functions.

`streamlit_app.py` stays a thin router (sidebar -> page dict -> call); each
module here owns exactly one nav destination's layout. Split out of what was
previously one large `streamlit_app.py` once the page count and per-page
complexity (tabs, cards) grew past what one file could hold reviewably.
"""

from __future__ import annotations
