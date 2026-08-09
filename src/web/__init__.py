"""Server-rendered HTML surfaces: the landing page and the query dashboard.

Templates and stylesheets rather than a second service. The API already
serves a landing page, the demo URL already works, and a Streamlit or SPA
front end would want its own container, its own CORS story and a second
deployment for a project whose whole point is one auditable request path.

Paths resolve from `__file__` so the same code works from a checkout and
from an installed package — `pip install .` in the Dockerfile puts `src`
into site-packages, and `pyproject.toml` declares the html/css/js as package
data so the installed copy is not a directory of nothing but `.py`.
"""

from pathlib import Path

from fastapi.templating import Jinja2Templates

WEB_DIR = Path(__file__).parent
TEMPLATE_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
