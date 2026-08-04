"""Small helper to load config.yaml from anywhere in the project."""
import os
import yaml

_CONFIG_CACHE = None


def _find_project_root(start: str) -> str:
    """Walk upwards until we find config.yaml."""
    current = os.path.abspath(start)
    for _ in range(6):
        if os.path.exists(os.path.join(current, "config.yaml")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Could not locate config.yaml in any parent directory.")


def get_project_root() -> str:
    return _find_project_root(os.path.dirname(__file__))


def load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    root = get_project_root()
    with open(os.path.join(root, "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = root
    _CONFIG_CACHE = cfg
    return cfg


def resolve_path(relative_path: str) -> str:
    root = get_project_root()
    full = os.path.join(root, relative_path)
    os.makedirs(os.path.dirname(full) if not full.endswith("/") else full, exist_ok=True)
    return full
