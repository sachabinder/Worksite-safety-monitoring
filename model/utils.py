import json
import os
import pathlib
from typing import List, Union


def get_filenames_of_path(path: pathlib.Path, ext: str = "*") -> List[pathlib.Path]:
    """
    Returns a list of files in a directory/path. Uses pathlib.
    """
    filenames = [file for file in path.glob(ext) if file.is_file()]
    assert len(filenames) > 0, f"No files found in path: {path}"
    return filenames


def read_json(path: pathlib.Path) -> dict:
    with open(str(path), "r") as fp:  # fp is the file pointer
        file = json.loads(s=fp.read())

    return file