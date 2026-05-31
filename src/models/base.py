from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.runnables import Runnable, RunnableConfig


class BaseModel(Runnable):
    """Abstract LCEL-compatible wrapper for a HuggingFace model.

    Subclasses MUST:

    - Set ``_config_key`` (class attribute, ``str``) to the key in the
      ``models`` section of the YAML config that holds their constructor
      keyword-arguments.
    - Implement ``__init__`` to load the HF model **once** at construction
      time.
    - Implement ``invoke`` to run the already-loaded model.

    The ``from_config`` classmethod reads the YAML file and passes the
    relevant section straight to ``__init__``.

    Example usage::

        sig = SigLIPModel.from_config("config.yaml")
        embedding = sig.invoke("a cat on a sofa")          # str  → np.ndarray
        chain = sig | RunnableLambda(lambda v: v.tolist())  # LCEL piping
    """

    #: Override in subclasses with the YAML section key, e.g. ``"siglip"``.
    _config_key: Optional[str] = None

    @classmethod
    def from_config(cls, path: str | Path = "config.yaml") -> "BaseModel":
        """Instantiate this model using values from *path* (a YAML file).

        The YAML must have a ``models.<_config_key>`` section whose keys
        match the constructor's keyword arguments::

            models:
              siglip:
                model_id: "google/siglip2-base-patch16-224"
                device: "auto"
                batch_size: 64

        Args:
            path: Path to the YAML configuration file.

        Raises:
            NotImplementedError: If the subclass has not set ``_config_key``.
            KeyError: If the expected section is missing from the YAML.
        """
        if cls._config_key is None:
            raise NotImplementedError(
                f"{cls.__name__} does not define _config_key; "
                "set it as a class attribute before calling from_config()."
            )
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(**cfg["models"][cls._config_key])

    @abstractmethod
    def invoke(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Run the model on *input* and return the result.

        This method is the single LCEL entry-point.  It is called by the
        LCEL machinery when the model is used in a chain (``model | step``).
        """
        ...
