import json
import requests
import zipfile
import base64
from io import BytesIO
from datetime import datetime
from .config import _RATE_LIMITED_API_URL, _CACHE_DIR, _CACHE_DATA_DIR, _CACHE_TABLE_NAMES_PATH, _CACHE_LAST_SYNCED_PATH

def _has_cache():
    """
    Checks if the cache is properly set up by ensuring that the cache
    directory and its necessary files and subdirectories exist and are non-empty.

    Specifically, it checks:
    - The existence of the main cache directory.
    - The existence of the 'tables' subdirectory.
    - The existence of the 'table_names.json' and 'last_synced.json' files.

    Returns:
        bool: True if the cache is fully set up (all directories and files exist and are non-empty),
              False otherwise.
    """
    if not _CACHE_DIR.exists() or not _CACHE_DIR.is_dir():
        return False

    if not _CACHE_DATA_DIR.exists() or not _CACHE_DATA_DIR.is_dir() or not any(_CACHE_DATA_DIR.iterdir()):
        return False

    if not _CACHE_TABLE_NAMES_PATH.exists() or not _CACHE_TABLE_NAMES_PATH.is_file():
        return False

    if not _CACHE_LAST_SYNCED_PATH.exists() or not _CACHE_LAST_SYNCED_PATH.is_file():
        return False

    return True

def _update_last_synced():
    """
    Record the current UTC time as the last successful data sync.

    This function creates or overwrites a local JSON file in the cache directory 
    with a timestamp indicating the most recent data refresh. The timestamp is 
    stored in ISO 8601 format.

    This file is used to support the `LAST_SYNCED` attribute in the public API.
    """
    _CACHE_LAST_SYNCED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_LAST_SYNCED_PATH, "w") as f:
        json.dump({"timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z"}, f)

def _cache_data_from_api():
    """
    Fetches a zip file of Parquet tables from the remote API,
    enforcing rate limiting. Extracts each Parquet file into the local cache 
    directory and saves the list of table names as JSON metadata.

    Raises:
        Exception: If the API call fails or the rate limit is exceeded.
    """
    _CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_TABLE_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)

    response = requests.post(_RATE_LIMITED_API_URL)

    if response.status_code == 429: 
        raise Exception(response.json()["message"])
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}")

    zip_content = base64.b64decode(response.text) 

    with zipfile.ZipFile(BytesIO(zip_content), "r") as zip_file:
        # Cache parquet files from the zip archive
        zip_file.extractall(_CACHE_DATA_DIR)

        # Cache table names
        table_names = [name.replace(".parquet", "") for name in zip_file.namelist()]
        with open(_CACHE_TABLE_NAMES_PATH, "w") as f:
            json.dump(table_names, f)



