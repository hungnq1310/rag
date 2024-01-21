from typing import Optional, TYPE_CHECKING
import os

from rag.utils.utils import get_cache_dir

from .huggingface import HuggingFaceEmbedding
from .mock import MockEmbedding

if TYPE_CHECKING:
    from rag.entity.embeddings.base_embeddings import EmbedType, BaseEmbedding

def resolve_embed_model(embed_model: Optional["EmbedType"] = None) -> "BaseEmbedding":
    """Resolve embed model."""

    if isinstance(embed_model, str):
        splits = embed_model.split(":", 1)
        model_name = splits[1] if len(splits) > 1 else splits

        cache_folder = os.path.join(get_cache_dir(), "models")
        os.makedirs(cache_folder, exist_ok=True)

        print("using mock embeddings!!")
        embed_model = HuggingFaceEmbedding(
            model_name=model_name, cache_folder=cache_folder
        )


    if embed_model is None:
        print("Embeddings have been explicitly disabled. Using MockEmbedding.")
        embed_model = MockEmbedding(embed_dim=1)

    return embed_model
