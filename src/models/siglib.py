from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from langchain_core.runnables import RunnableConfig
from transformers import AutoModel, AutoProcessor

from .base import BaseModel


class SigLIPModel(BaseModel):
    """LCEL-compatible wrapper for SigLIP2 (``google/siglip2-base-patch16-224``).

    Loads the SigLIP2 vision-language model once at construction time.
    ``invoke`` dispatches to the appropriate embedding method based on the
    input type:

    - ``str``                    → :meth:`embed_text`   → 1-D ``(D,)`` array
    - ``PIL.Image.Image``        → :meth:`embed_images` → 1-D ``(D,)`` array
    - ``list[PIL.Image.Image]``  → :meth:`embed_images` → 2-D ``(N, D)`` array

    All returned vectors are L2-normalised.

    Example::

        sig = SigLIPModel.from_config("config.yaml")
        v = sig.invoke("a laptop on a desk")          # (768,) ndarray
        chain = sig | RunnableLambda(lambda v: v.tolist())
    """

    _config_key = "siglip"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        """Load the SigLIP2 model and processor.

        The model is loaded **once here**; ``invoke`` never reloads it.

        Args:
            model_id:   HuggingFace model identifier.
            device:     Device string forwarded to HF as ``device_map``
                        (``"auto"``, ``"cuda"``, ``"cpu"``).
            batch_size: Stored for use by callers that want to batch images
                        before calling :meth:`embed_images`.
        """
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, device_map=device).eval()
        self.device = device
        self.batch_size = batch_size
        self.embedding_dim: int = self.model.config.text_config.hidden_size

    # ------------------------------------------------------------------
    # Embedding helpers
    # (bodies kept verbatim from notebooks/SigLib.ipynb embed-helpers cell)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed_images(self, pil_images: list) -> np.ndarray:
        """Embed a batch of PIL images.

        Args:
            pil_images: List of ``PIL.Image.Image`` objects.

        Returns:
            Float32 numpy array of shape ``(N, embedding_dim)``,
            each row L2-normalised.
        """
        inp = self.processor(images=pil_images, return_tensors="pt").to(self.model.device)
        vecs = self.model.vision_model(**inp).pooler_output
        return torch.nn.functional.normalize(vecs, dim=-1).cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: The query / caption to embed.

        Returns:
            Float32 numpy array of shape ``(embedding_dim,)``,
            L2-normalised.
        """
        # SigLIP requires padding="max_length" for text inputs
        inp = self.processor(
            text=[text], return_tensors="pt", padding="max_length"
        ).to(self.model.device)
        vec = self.model.text_model(**inp).pooler_output
        return torch.nn.functional.normalize(vec, dim=-1).cpu().numpy()[0]

    # ------------------------------------------------------------------
    # LCEL entry point
    # ------------------------------------------------------------------

    def invoke(
        self,
        input: Union[str, Image.Image, list],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run SigLIP2 on *input* and return a normalised embedding as a dict.

        Dispatches on type:

        - ``str``                    → :meth:`embed_text`
        - ``PIL.Image.Image``        → :meth:`embed_images` (1-D vector)
        - ``list[PIL.Image.Image]``  → :meth:`embed_images` (2-D batch)

        Args:
            input:  The item to embed.
            config: Optional LangChain run configuration (unused here).

        Returns:
            A dict with a single key:

            - ``"embedding"`` — L2-normalised ``np.ndarray`` of shape
              ``(D,)`` for a single string or image, or ``(N, D)`` for a
              list of images.

        Raises:
            TypeError: For any other input type.
        """
        if isinstance(input, str):
            return {"embedding": self.embed_text(input)}
        if isinstance(input, Image.Image):
            return {"embedding": self.embed_images([input])[0]}
        if isinstance(input, list) and all(isinstance(x, Image.Image) for x in input):
            return {"embedding": self.embed_images(input)}
        raise TypeError(
            f"SigLIPModel.invoke: unsupported input type {type(input).__name__!r}. "
            "Expected str, PIL.Image.Image, or list[PIL.Image.Image]."
        )
