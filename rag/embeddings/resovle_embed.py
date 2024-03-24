from typing import Optional, TYPE_CHECKING
import os

from rag.rag_utils.utils import get_cache_dir
from .huggingface import HuggingFaceEmbedding
from .mock import MockEmbedding

if TYPE_CHECKING:
    from .base_embeddings import EmbedType, BaseEmbedding


def resolve_embed_model(embed_model: Optional["EmbedType"] = None) -> "BaseEmbedding":
    """Resolve embed model."""

    if isinstance(embed_model, str):
        # cache
        cache_folder = os.path.join(get_cache_dir(), "models")
        os.makedirs(cache_folder, exist_ok=True)
        
        print("using mock embeddings!!")
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model, cache_folder=cache_folder
        )


    if embed_model is None:
        print("Embeddings have been explicitly disabled. Using MockEmbedding.")
        embed_model = MockEmbedding(embed_dim=1)

    return embed_model
