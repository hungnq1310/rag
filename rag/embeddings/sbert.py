import asyncio
from typing import Any, List, Optional, Dict

from rag.bridge.pydantic import Field, PrivateAttr
from rag.callbacks import CallbackManager
from rag.constants.default_huggingface import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL as SENTENCE_TRANSFOERMER_MODEL
from rag.rag_utils.utils import get_cache_dir, infer_torch_device

from .base_embeddings import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
)
from .utils import (
    format_query,
    format_text,
)

DEFAULT_HUGGINGFACE_LENGTH = 512


class SBertEmbedding(BaseEmbedding):
   
    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )

    prompts : Dict[str, str] = Field(
        description="""A dictionary with prompts for the model. The key is the prompt name, the value is the prompt text. 
        For example: {“query”: “query: “, “passage”: “passage: “}"""
    )

    default_prompt_name : str = Field(
        description="The name of the prompt that should be used by default. If not set, no prompt will be applied."
    )
    
    cache_folder: Optional[str] = Field(
        description="Cache folder for huggingface files."
    )
    device: str = Field(
        description="device to load model. Default to `cpu`"
    )
    
    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        max_length: Optional[int] = None,
        token: Optional[str] = None,
        prompts : Optional[Dict[str, str]] = None,
        default_prompt_name : Optional[str] = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        cache_folder: Optional[str] = None,
        device: Optional[str] = None,
        callback_manager: CallbackManager = CallbackManager([]),
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "SentenceTransformer requires sentence_transformers to be installed.\n"
                "Please install transformers with `pip install sentence_transformers.`"
            )

        if model_name_or_path is None:  # Use model_name with AutoModel
            model_name_or_path = SENTENCE_TRANSFOERMER_MODEL
            print(f"Using default model: {model_name_or_path}")

        if max_length is None:
            try:
                max_length = int(self._model.config.max_position_embeddings)
            except AttributeError as exc:
                raise ValueError(
                    "Unable to find max_length from model config. Please specify max_length."
                ) from exc

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            model_name=model_name_or_path,
        )

        # set attributes
        self.device = device or infer_torch_device()
        self.cache_folder = cache_folder or get_cache_dir()
        self.max_length = max_length or DEFAULT_HUGGINGFACE_LENGTH
        self.prompts = prompts or {}
        self.default_prompt_name = default_prompt_name or ""
        
        # set private attribute
        model = SentenceTransformer(
            model_name_or_path, cache_folder=cache_folder, use_auth_token=token,
        )
        self._model = model.to(self.device)


    def get_model_dim(self) -> int:
        """Get model."""
        if self._model is None:
            raise ValueError("Model is not loaded yet.")
        return self._model.config.hidden_size

    @classmethod
    def class_name(cls) -> str:
        return "SBertEmbedding"

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""

        embeddings = self._model.encode(sentences)
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query_instruction = self.prompts.get("query", "")
        query = format_query(query, self.model_name, query_instruction)
        return self._embed([query])[0]
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        passage_instruction = self.prompts.get("passage", "")
        query = format_query(text, self.model_name, passage_instruction)
        return self._embed([text])[0]


    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        passage_instruction = self.prompts.get("passage", "")
        texts = [
            format_text(text, self.model_name, passage_instruction) for text in texts
        ]
        return self._embed(texts)