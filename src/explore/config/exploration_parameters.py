import os
from typing import Any, Dict

import torch
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"


def _validate_type(name: str, value: Any, expected_type: type):
    if not isinstance(value, expected_type):
        raise TypeError(
            f"Parameter '{name}' must be of type {expected_type.__name__}, got {type(value).__name__}"
        )


def _validate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    _validate_type("model_path", params.get("model_path"), str)
    return params


def load_params() -> Dict[str, Any]:
    """
    Load the parameters for exploration from the YAML config file, with validation and typing.
    """
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    with open(config_path, "r") as f:
        parameters = yaml.safe_load(f)
    return _validate_params(parameters)


explore_params = load_params()
