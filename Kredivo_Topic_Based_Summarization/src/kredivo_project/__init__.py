import os
import tempfile

if "NUMBA_CACHE_DIR" not in os.environ:
    os.environ["NUMBA_CACHE_DIR"] = os.path.join(tempfile.gettempdir(), "numba_cache")

"""Topic-Based, Evidence-Grounded Multi-Document Extractive Summarization."""
__version__ = "1.0.0"
