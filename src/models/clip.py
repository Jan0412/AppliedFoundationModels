from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from PIL import Image
from langchain_core.runnables import RunnableConfig
from transformers import AutoModel, AutoProcessor

from .base import BaseModel


class CLIPEmbedModel(BaseModel):
    """LCEL-compatible wrapper for OpenAI CLIP (``openai/clip-vit-base-patch16``).

    Drop-in alternative to :class:`~src.models.siglib.SigLIPModel`: identical
    ``embed_text`` / ``embed_images`` / ``invoke`` contract, identical
    L2-normalised output, so the indexer and the query pipeline never learn
    which of the two is wired (see :func:`src.models.factory.load_embedder`).

    Two CLIP-specific details leak through the shared contract:

    - :attr:`embedding_dim` is CLIP's **projection** dim (512 for ViT-B/16),
      not an encoder hidden size. The image and text towers have different
      hidden sizes (768 / 512) and only meet after their projection heads, so
      the projected vectors are the ones that get stored and compared.
    - CLIP's text encoder is capped at 77 tokens, so :meth:`embed_text`
      truncates long queries instead of raising. Full ScanRefer descriptions
      can exceed that; category-noun queries never do.

    Vectors are **not** interchangeable with SigLIP's — different dimension and
    a different space — so a collection indexed with one embedder cannot be
    searched with the other. Each therefore gets its own LanceDB store; see
    ``indexing.db_paths`` in ``config.yaml``.

    Example::

        clip = CLIPEmbedModel.from_config("config.yaml")
        v = clip.invoke("a laptop on a desk")["embedding"]   # (512,) ndarray
    """

    _config_key = "clip"

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        """Load the CLIP model and processor.

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
        # Projection dim, not hidden size: see the class docstring.
        self.embedding_dim: int = self.model.config.projection_dim

    # ------------------------------------------------------------------
    # Embedding helpers
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
        # ``get_image_features`` returns the vision output with ``pooler_output``
        # replaced by the *projected* embedding — the shared CLIP space.
        vecs = self.model.get_image_features(**inp).pooler_output
        return torch.nn.functional.normalize(vecs, dim=-1).cpu().numpy()

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: The query / caption to embed. Truncated at CLIP's 77-token
                  context limit.

        Returns:
            Float32 numpy array of shape ``(embedding_dim,)``,
            L2-normalised.
        """
        # CLIP pads to the longest sequence in the batch (unlike SigLIP, which
        # requires padding="max_length"), and truncates at its 77-token cap.
        inp = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)
        vec = self.model.get_text_features(**inp).pooler_output
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
        """Run CLIP on *input* and return a normalised embedding as a dict.

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
            f"CLIPEmbedModel.invoke: unsupported input type {type(input).__name__!r}. "
            "Expected str, PIL.Image.Image, or list[PIL.Image.Image]."
        )
