import json
import pandas as pd
import requests
import base64
from .sync import _cache_data_from_api, _update_last_synced, _has_cache
from .config import _CACHE_DATA_DIR, _CACHE_TABLE_NAMES_PATH, _CACHE_LAST_SYNCED_PATH

def refresh_data(verbose=False):
    """
    Refresh the local data cache by downloading the latest available datasets from the source.

    This function updates all Survivor datasets and the list of available table names.
    It should be used when you want to ensure you have the most current version of the data.

    After calling this function, you can access updated datasets using `load("table_name")` 
    or module-level attributes like `survivorpy.castaways`.

    Args:
        verbose (bool): If True, prints a summary of what changed during the refresh.

    Example:
        from survivorpy import refresh_data
        refresh_data(verbose=True)
    """
    update_info = _get_data_update_info()
    last_updated_remote = update_info["timestamp"]
    has_cache = _has_cache()
    if has_cache:
        last_synced_local = get_last_synced()
    if verbose:
        if has_cache:
            print(f"Local data cache was last synced on:  {last_synced_local}")
        print(f"Latest available data was updated on: {last_updated_remote}")

    if not has_cache or (last_synced_local < last_updated_remote):
        _cache_data_from_api()
        _update_last_synced()
        if verbose:
            changes = []
            if update_info.get("added"):
                changes.append(f"Tables added:    {update_info['added']}")
            if update_info.get("modified"):
                changes.append(f"Tables modified: {update_info['modified']}")
            if update_info.get("deleted"):
                changes.append(f"Tables deleted:  {update_info['deleted']}")

            if changes:
                print("Summary of changes from most recent update:")
                for line in changes:
                    print(f"    - {line}")
            else:
                print("No changes detected in the data.")
    else:
        if verbose:
            print("Local data cache is already up to date.")

def load(table: str) -> pd.DataFrame:
    """
    Load a Survivor dataset from the local cache.

    This function reads a previously cached dataset into a pandas DataFrame. 
    It does not attempt to download or refresh the data. Use `refresh_data()` 
    first if you need to ensure the local data is up to date.

    Parameters:
        table (str): The name of the dataset to load (e.g., "castaways").

    Returns:
        pd.DataFrame: The requested dataset.

    Raises:
        ValueError: If the provided table name is not recognized.
        OSError, ValueError, etc.: If the local file is missing or unreadable.

    Example:
        import survivorpy as sv
        df = sv.load("castaways")
    """
    table_names_list = get_table_names()

    if table not in table_names_list:
        raise ValueError(f"Unknown table: '{table}'. Choose from: {table_names_list}")

    local_path = _CACHE_DATA_DIR / f"{table}.parquet"
    return pd.read_parquet(local_path)

def get_table_names():
    """
    Load the list of available Survivor datasets from the local cache.

    This function reads the names of previously cached datasets. It does not attempt 
    to download or refresh the data. Use `refresh_data()` to update the cache 
    if you need to ensure the list of datasets is current.

    Returns:
        list[str]: A list of dataset names (e.g., [..., "castaways",...]).

    Raises:
        OSError: If the table names cache file is missing or unreadable.

    Example:
        import survivorpy as sv
        tables = sv.get_table_names()

    Notes:
        You can also access the available table names via the `TABLE_NAMES` attribute.
    """
    with open(_CACHE_TABLE_NAMES_PATH, "r") as f:
        return json.load(f)

def get_last_synced():
    """
    Return the timestamp of the most recent successful data sync.

    Returns:
        str: An ISO 8601 timestamp string (e.g., "2025-04-25T19:12:05.123Z").

    Raises:
        FileNotFoundError: If the sync timestamp file does not exist.
        ValueError: If the file contents are invalid.

    Example:
        import survivorpy as sv
        tables = sv.get_last_synced()

    Notes:
        You can also access the available table names via the `LAST_SYNCED` attribute.

    """
    with open(_CACHE_LAST_SYNCED_PATH, "r") as f:
        return json.load(f)["timestamp"]

def _get_data_update_info():
    """
    Fetch metadata about the most recent data update from GitHub.

    Returns:
        dict: A dictionary with keys 'added', 'deleted', 'modified', and 'timestamp'.

    Raises:
        Exception: If the GitHub API request fails or the content can't be parsed.
    """
    url = "https://api.github.com/repos/jonnycomes/survivorpy/contents/data_pipeline/metadata/data_last_updated.json"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        raise Exception(f"GitHub API request failed (status code {response.status_code}).")

    try:
        content = response.json()["content"]
        decoded = base64.b64decode(content).decode("utf-8")
        full_data = eval(decoded)
        return {k: v for k, v in full_data.items() if k != "hashes"}
    except Exception as e:
        raise Exception(f"Failed to parse GitHub API content: {e}")

