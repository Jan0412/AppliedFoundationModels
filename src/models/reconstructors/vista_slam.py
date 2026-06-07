import subprocess
import yaml
from pathlib import Path
from .base import BaseReconstructor

class ViSTASLAMReconstructor(BaseReconstructor):
    _config_key = "vista_slam"

    def __init__(self, **kwargs):
        """Initializes the wrapper and serializes the backend configuration."""
        # Separate runtime-specific parameters from the configuration dictionary
        self.output_dir_default = kwargs.pop("output_dir", "data/vista_slam_output")
        self.config_params = kwargs
        
        self.submodule_path = Path("external/vista-slam")
        self.submodule_config_path = self.submodule_path / "configs/custom_3DRoomSearch.yaml"
        
        self._generate_backend_config()

    def _generate_backend_config(self) -> None:
        """Writes the custom configuration parameters to the expected submodule path."""
        self.submodule_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.submodule_config_path, "w") as f:
            yaml.dump(self.config_params, f, default_flow_style=False)

    def invoke(self, input_dir: Path, output_dir: Path) -> None:
        """Triggers the external ViSTA-SLAM pipeline via an isolated subprocess."""
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
            
        # 1. Resolve absolute paths to survive the cwd change in the subprocess execution context
        abs_input = input_dir.resolve()
        abs_output = output_dir.resolve()
        
        abs_output.mkdir(parents=True, exist_ok=True)

        # 2. Append the required glob pattern for the ViSTA-SLAM data loader
        image_pattern = f"{abs_input}/*.png"

        # 3. Assemble the explicit CLI execution command array
        cmd = [
            "uv", "run", "python", "run.py",
            "--config", str(self.submodule_config_path.relative_to(self.submodule_path)),
            "--images", image_pattern,
            "--output", str(abs_output)
        ]

        # 4. Block on execution and stream terminal pipes directly to parent context descriptors
        subprocess.run(
            cmd, 
            cwd=str(self.submodule_path), 
            check=True, 
            stdout=None, 
            stderr=None
        )