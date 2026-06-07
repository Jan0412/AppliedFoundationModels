import yaml
from abc import ABC, abstractmethod
from pathlib import Path

class BaseReconstructor(ABC):
    """Abstract base class for uncalibrated sequence-level spatial reconstruction."""
    
    _config_key: str = ""

    @classmethod
    def from_config(cls, config_path: str | Path = "config.yaml") -> "BaseReconstructor":
        if not cls._config_key:
            raise NotImplementedError(f"{cls.__name__} must define a valid non-empty _config_key.")
            
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        try:
            model_kwargs = cfg["models"]["reconstructors"][cls._config_key]
        except KeyError as e:
            raise KeyError(f"Missing section ['models']['reconstructors']['{cls._config_key}'] in {config_path}") from e
            
        return cls(**model_kwargs)

    @abstractmethod
    def invoke(self, input_dir: Path, output_dir: Path) -> None:
        """Executes the reconstruction pipeline using input frames from input_dir and saving to output_dir."""
        pass