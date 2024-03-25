import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

from rag.bridge.pydantic import Field, PrivateAttr
from rag.callbacks import CallbackManager
from rag.constants.default_huggingface import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
from rag.rag_utils.utils import get_cache_dir, infer_torch_device
#TODO: change rag_utils to genral_utils

from .base_embeddings import (
    DEFAULT_EMBED_BATCH_SIZE,
    BaseEmbedding,
    Embedding
)
from .pooling import Pooling
from .utils import (
    format_query,
    format_text,
)

if TYPE_CHECKING:
    import torch

DEFAULT_HUGGINGFACE_LENGTH = 512


class HuggingFaceEmbedding(BaseEmbedding):
    max_length: int = Field(
        default=DEFAULT_HUGGINGFACE_LENGTH, description="Maximum length of input.", gt=0
    )
    pooling: Pooling = Field(default=Pooling.CLS, description="Pooling strategy.")
    normalize: bool = Field(default=True, description="Normalize embeddings or not.")
    query_instruction: Optional[str] = Field(
        description="Instruction to prepend to query text."
    )
    text_instruction: Optional[str] = Field(
        description="Instruction to prepend to text."
    )
    cache_folder: Optional[str] = Field(
        description="Cache folder for huggingface files."
    )
    device: str = Field(
        description="device to load model. Default to `cpu`"
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        pooling: Union[str, Pooling] = "cls",
        max_length: Optional[int] = None,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        normalize: bool = True,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        token: Optional[str] = None,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        device: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "HuggingFaceEmbedding requires transformers to be installed.\n"
                "Please install transformers with `pip install transformers`."
            )

        # set attribute
        self.device = device or infer_torch_device()
        self.cache_folder = cache_folder or get_cache_dir()
        self.normalize = normalize
        self.query_instruction = query_instruction
        self.text_instruction = text_instruction
        
        if model_name is None:  # Use model_name with AutoModel
            model_name = DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            print(f"Using default model: {model_name}")

        if tokenizer_name is None:  # Use tokenizer_name with AutoTokenizer
            tokenizer_name = model_name or DEFAULT_HUGGINGFACE_EMBEDDING_MODEL
            print(f"Using default tokenizer of model: {model_name}")

        if max_length is None:
            try:
                self.max_length = int(self._model.config.max_position_embeddings)
            except AttributeError as exc:
                raise ValueError(
                    "Unable to find max_length from model config. Please specify max_length."
                ) from exc

        if isinstance(pooling, str):
            try:
                self.pooling = Pooling(pooling)
            except ValueError as exc:
                raise NotImplementedError(
                    f"Pooling {pooling} unsupported, please pick one in"
                    f" {[p.value for p in Pooling]}."
                ) from exc

        super().__init__(
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager or CallbackManager(),
            model_name=model_name,
        )
        # set private attribute
        model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_folder, trust_remote_code=trust_remote_code, token=token,
        )
        self._model = model.to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_folder
        )
        self._tokenizer = tokenizer

    def get_model_dim(self) -> int:
        """Get model."""
        if self._model is None:
            raise ValueError("Model is not loaded yet.")
        return self._model.config.hidden_size

    @classmethod
    def class_name(cls) -> str:
        return "HuggingFaceEmbedding"

    def _mean_pooling(
        self, token_embeddings: "torch.Tensor", attention_mask: "torch.Tensor"
    ) -> "torch.Tensor":
        """Mean Pooling - Take attention mask into account for correct averaging."""
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        numerator = (token_embeddings * input_mask_expanded).sum(1)
        return numerator / input_mask_expanded.sum(1).clamp(min=1e-9)

    def _embed(self, sentences: List[str]) -> List[List[float]]:
        """Embed sentences."""
        encoded_input = self._tokenizer(
            sentences,
            padding=True,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # move tokenizer inputs to device
        encoded_input = {
            key: val.to(self.device) for key, val in encoded_input.items()
        }

        model_output = self._model(**encoded_input)

        if self.pooling == Pooling.CLS:
            context_layer: "torch.Tensor" = model_output[0]
            embeddings = self.pooling.cls_pooling(context_layer)
        else:
            embeddings = self._mean_pooling(
                token_embeddings=model_output[0],
                attention_mask=encoded_input["attention_mask"],
            )

        if self.normalize:
            import torch

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        query = format_query(query, self.model_name, self.query_instruction)
        return self._embed([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding async."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding async."""
        return self._get_text_embedding(text)

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        text = format_text(text, self.model_name, self.text_instruction)
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        texts = [
            format_text(text, self.model_name, self.text_instruction) for text in texts
        ]
        return self._embed(texts)
