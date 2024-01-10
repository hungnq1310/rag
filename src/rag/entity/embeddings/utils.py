"""Embedding utils for LlamaIndex."""
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from llama_index.bridge.langchain import Embeddings as LCEmbeddings
from .base_embeddings import BaseEmbedding

EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]


def save_embedding(embedding: List[float], file_path: str) -> None:
    """Save embedding to file."""
    with open(file_path, "w") as f:
        f.write(",".join([str(x) for x in embedding]))


def load_embedding(file_path: str) -> List[float]:
    """Load embedding from file. Will only return first embedding in file."""
    with open(file_path) as f:
        for line in f:
            embedding = [float(x) for x in line.strip().split(",")]
            break
        return embedding
