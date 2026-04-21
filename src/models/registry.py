import importlib
from dataclasses import dataclass


@dataclass
class ModelEntry:
    name:   str
    cls:    type
    module: str = ""
 
    def __post_init__(self):
        if not self.module:
            self.module = self.cls.__module__   # e.g. "src.models.rnn"


class ModelRegistry:
    """Class that assosiates model name with the model module and class."""
    def __init__(self):
        self._registry: dict[str, ModelEntry] = {}

    def register(self, name: str):
        def decorator(cls):
            self._registry[name] = ModelEntry(name=name, cls=cls)
            cls.model_name = name
            return cls
        return decorator
    
    def build(self, name: str, config: dict):
        """
        Instantiate a model by name.
        Imports the module on demand so @register fires if not already loaded.
        """
        if name not in self._registry:
            importlib.import_module(f"src.models.{name}")   # triggers @register
 
        entry = self.get(name)
        return entry.cls(config)

    def get(self, name: str) -> ModelEntry:
        if name not in self._registry:
            raise KeyError(f"Model '{name}' not found in the registry.")
        return self._registry[name]
    
registry = ModelRegistry()    

