import os,sys
from box.exceptions import BoxValueError
import yaml
from src.textSummarizer.logging import logger
from box import ConfigBox
from pathlib import Path
from typing import Any
from ensure import ensure_annotations

@ensure_annotations
def read_yaml(yaml_path:Path) -> ConfigBox:
    try:
        logger.info(f"entered read_yaml function at {os.getcwd()}")
        with open(yaml_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {yaml_path} loaded")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        logger.exception(e)
        raise e
    
@ensure_annotations
def create_directories(path_to_directories:list, verbose=True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at {path}")
