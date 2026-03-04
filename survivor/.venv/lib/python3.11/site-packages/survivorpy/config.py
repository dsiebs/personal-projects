from pathlib import Path
from appdirs import user_cache_dir

_CACHE_DIR = Path(user_cache_dir("survivorpy", "jonnycomes"))
_CACHE_DATA_DIR = _CACHE_DIR / "tables"
_CACHE_TABLE_NAMES_PATH = _CACHE_DIR / "table_names.json"
_CACHE_LAST_SYNCED_PATH = _CACHE_DIR / "last_synced.json"

_RATE_LIMITED_API_URL = "https://xtz4x23wzg.execute-api.us-west-2.amazonaws.com/prod/rate-limit"