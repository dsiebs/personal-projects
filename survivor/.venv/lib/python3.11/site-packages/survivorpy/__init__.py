"""
SurvivorPy is a Python wrapper for the data provided by the R package 'survivoR'. 
It allows users to easily access clean, structured data on *Survivor* contestants, 
seasons, episodes, votes, and other related content, all through Python and in the 
form of `pandas` DataFrames.

Key Features:
- Seamless integration with the R package 'survivoR', providing up-to-date data (with no more than a 24 hour delay).
- Direct access to *Survivor* data tables via the `load()` function or direct table imports.
- Efficient caching mechanism that allows for offline access to tables while syncing data with the source.
- Includes attributes like `TABLE_NAMES` for listing available data tables and `LAST_SYNCED` for tracking data synchronization.

Data is cached upon first access, and users can refresh their local cache to ensure they have the latest available data by calling the `refresh_data()` function.

This module provides an easy and efficient way to interact with detailed *Survivor* data, designed to integrate smoothly with the Python data ecosystem.

"""

from .data import load, refresh_data, get_table_names, get_last_synced
from .sync import _has_cache

# Refresh the data if cache is not set up
if not _has_cache():
    refresh_data()

# Fetch key attributes for module access
TABLE_NAMES = get_table_names()
LAST_SYNCED = get_last_synced()

# Define the public API
__all__ = ["load", 
           "refresh_data", 
           "TABLE_NAMES", "get_table_names", 
           "LAST_SYNCED", "get_last_synced"
           ] + TABLE_NAMES

def __getattr__(name):
    """
    Dynamically load tables when accessed as attributes.
    Raises AttributeError if the table name is not found.
    """
    if name in get_table_names():
        return load(name)
    raise AttributeError(
        f"module 'survivorpy' has no attribute '{name}'. "
        f"Available tables: {', '.join(get_table_names())}"
    )
