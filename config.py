import os
from pathlib import Path


def load_env(path=Path(__file__).with_name(".env")):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def getenv(name, *fallbacks, default=None):
    for key in (name, *fallbacks):
        value = os.getenv(key)
        if value:
            return value
    return default


load_env()

LLM_PROVIDER = getenv("LLM_PROVIDER", default="openai-compatible")
API_KEY = getenv("LLM_API_KEY", "OPENAI_API_KEY")
MODEL = getenv("LLM_MODEL", "OPENAI_MODEL", default="deepseek-flash")
BASE_URL = getenv("LLM_BASE_URL", "OPENAI_BASE_URL", default="https://api.deepseek.com").rstrip("/")
CHAT_COMPLETIONS_URL = getenv("LLM_CHAT_COMPLETIONS_URL", default=f"{BASE_URL}/chat/completions")
MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))
SHOW_PROVIDER_META = os.getenv("SHOW_PROVIDER_META", "1") != "0"
