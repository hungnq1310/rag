import os 
from box.exceptions import BoxValueError
import yaml
from logging import getLogger
from ensure import ensure_annotations
from box import ConfigBox 
from pathlib import Path
from typing import Any

logger = getLogger(__name__)

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox: 
    """Read yaml file and returns ConfigBox instance

    Args:
        path_to_yaml: Path to yaml file
    
    Releases:
        ValueError: if yaml file is empty 
        e: empty file
    Returns:
        ConfigBox instance
    """
    try: 
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"yaml file: {path_to_yaml} is empty")
    except Exception as e: 
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True): 
    """Create list of directories 

    Args: 
        path_to_directories: list of path of directories 
        verbose: bool, if True, print information about created directories
    """
    for path in path_to_directories: 
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"directory: {path} created successfully")

@ensure_annotations
def get_size(path: Path) -> int: 
    """Get size in Kbytes
    
    Args: 
        path: Path to file
    
    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return int(size_in_kb)