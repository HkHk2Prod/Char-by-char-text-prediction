import importlib
import pkgutil
from pathlib import Path


def autodiscover():
    models_dir = str(Path(__file__).parent)
    for _, module_name, _ in pkgutil.iter_modules([models_dir]):
        if module_name not in ("base", "registry"):
            importlib.import_module(f"src.models.{module_name}")

