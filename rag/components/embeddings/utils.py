from typing import Optional, TYPE_CHECKING
import os

from rag.constants.default_huggingface import (
    INSTRUCTOR_MODELS,
    DEFAULT_QUERY_INSTRUCTION,
    BGE_MODELS,
    DEFAULT_QUERY_BGE_INSTRUCTION_ZH,
    DEFAULT_EMBED_INSTRUCTION,
    DEFAULT_QUERY_BGE_INSTRUCTION_EN
)
from rag.utils.utils import get_cache_dir

from .huggingface import HuggingFaceEmbedding
from .mock import MockEmbedding

if TYPE_CHECKING:
    from rag.entity.embeddings import EmbedType, BaseEmbedding

def get_query_instruct_for_model_name(model_name: Optional[str]) -> str:
    """Get query text instruction for a given model name."""
    if model_name in INSTRUCTOR_MODELS:
        return DEFAULT_QUERY_INSTRUCTION
    if model_name in BGE_MODELS:
        if "zh" in model_name:
            return DEFAULT_QUERY_BGE_INSTRUCTION_ZH
        return DEFAULT_QUERY_BGE_INSTRUCTION_EN
    return ""


def format_query(
    query: str, model_name: Optional[str], instruction: Optional[str] = None
) -> str:
    if instruction is None:
        instruction = get_query_instruct_for_model_name(model_name)
    # NOTE: strip() enables backdoor for defeating instruction prepend by
    # passing empty string
    return f"{instruction} {query}".strip()


def get_text_instruct_for_model_name(model_name: Optional[str]) -> str:
    """Get text instruction for a given model name."""
    return DEFAULT_EMBED_INSTRUCTION if model_name in INSTRUCTOR_MODELS else ""


def format_text(
    text: str, model_name: Optional[str], instruction: Optional[str] = None
) -> str:
    if instruction is None:
        instruction = get_text_instruct_for_model_name(model_name)
    # NOTE: strip() enables backdoor for defeating instruction prepend by
    # passing empty string
    return f"{instruction} {text}".strip()



def resolve_embed_model(embed_model: Optional["EmbedType"] = None) -> "BaseEmbedding":
    """Resolve embed model."""

    if isinstance(embed_model, str):
        splits = embed_model.split(":", 1)
        is_local = splits[0]
        model_name = splits[1] if len(splits) > 1 else None
        if is_local != "local":
            raise ValueError(
                "embed_model must start with str 'local' or of type BaseEmbedding"
            )

        cache_folder = os.path.join(get_cache_dir(), "models")
        os.makedirs(cache_folder, exist_ok=True)

        embed_model = HuggingFaceEmbedding(
            model_name=model_name, cache_folder=cache_folder
        )


    if embed_model is None:
        print("Embeddings have been explicitly disabled. Using MockEmbedding.")
        embed_model = MockEmbedding(embed_dim=1)

    return embed_model
