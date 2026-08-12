import os

_HTML_PATH = os.path.join(os.path.dirname(__file__), "frontend.html")

def get_frontend_html() -> str:
    with open(_HTML_PATH, "r", encoding="utf-8") as f:
        return f.read()
