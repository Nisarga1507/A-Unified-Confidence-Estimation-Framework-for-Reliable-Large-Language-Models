"""
Shared utility functions used throughout the project.
"""

import os
from pathlib import Path


def create_directory(path):
    """
    Create a directory if it does not already exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def file_exists(path):
    """
    Check whether a file exists.
    """
    return Path(path).exists()
