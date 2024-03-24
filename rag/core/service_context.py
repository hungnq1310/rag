import logging
from typing import List, Optional, TYPE_CHECKING


from rag.bridge.pydantic import BaseModel
from rag.schema.component import TransformComponent
from rag.callbacks.callback_manager import CallbackManager
from rag.llm.resolve_llm import resolve_llm
from rag.embeddings.resovle_embed import resolve_embed_model
from rag.node_parser.text.sentence import (
    DEFAULT_CHUNK_SIZE,
    SENTENCE_CHUNK_OVERLAP,
    SentenceSplitter,
)
from rag.embeddings.base_embeddings import BaseEmbedding, EmbedType
from rag.llm.base import LLM, LLMType
from rag.node_parser.base import NodeParser, TextSplitter


logger = logging.getLogger(__name__)


def _get_default_node_parser(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = SENTENCE_CHUNK_OVERLAP,
    callback_manager: Optional[CallbackManager] = None,
) -> SentenceSplitter:
    """Get default node parser."""
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        callback_manager=callback_manager or CallbackManager(),
    )


class ServiceContextData(BaseModel):
    llm: dict
    embed_model: dict
    transformations: List[dict]


class ServiceContext:
    """Service Context container.

    The service context container is a utility container for LlamaIndex
    index and query classes. It contains the following:
    - prompt_helper: PromptHelper
    - embed_model: BaseEmbedding
    - node_parser: TextNodesParser
    - callback_manager: CallbackManager

    """
    _instance = None

    llm: LLM
    embed_model: BaseEmbedding
    transformations: List[TransformComponent]
    callback_manager: CallbackManager

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self,
        llm: Optional[LLMType] = None,
        embed_model: Optional[EmbedType] = None,
        node_parser: Optional[NodeParser] = None,
        text_splitter: Optional[TextSplitter] = None,
        transformations: Optional[List[TransformComponent]] = None,
        callback_manager: Optional[CallbackManager] = None,
        # node parser kwargs
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        """
        Args:
            prompt_helper (Optional[PromptHelper]): PromptHelper
            embed_model (Optional[BaseEmbedding]): BaseEmbedding
                or "local" (use local model)
            node_parser (Optional[TextNodesParser]): TextNodesParser
            chunk_size (Optional[int]): chunk_size
            callback_manager (Optional[CallbackManager]): CallbackManager
            system_prompt (Optional[str]): System-wide prompt to be prepended
                to all input prompts, used to guide system "decision making"

        """
        self.callback_manager = callback_manager or CallbackManager([])
        self.llm = resolve_llm(llm)

        # NOTE: the embed_model isn't used in all indices
        # NOTE: embed model should be a transformation, but the way the service
        # context works, we can't put in there yet.
        self.embed_model = resolve_embed_model(embed_model)
        self.embed_model.callback_manager = self.callback_manager

        if text_splitter is not None and node_parser is not None:
            raise ValueError("Cannot specify both text_splitter and node_parser")

        # text splitter extends node parser
        node_parser = text_splitter \
            or node_parser \
            or _get_default_node_parser(
                chunk_size=chunk_size or DEFAULT_CHUNK_SIZE,
                chunk_overlap=chunk_overlap or SENTENCE_CHUNK_OVERLAP,
                callback_manager=self.callback_manager,
            )

        self.transformations = transformations or [node_parser]

        
    @classmethod
    def from_service_context(
        cls,
        service_context: "ServiceContext",
        llm: Optional[LLMType] = "default",
        embed_model: Optional[EmbedType] = "default",
        node_parser: Optional[NodeParser] = None,
        text_splitter: Optional[TextSplitter] = None,
        transformations: Optional[List[TransformComponent]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ) -> "ServiceContext":
        """Instantiate a new service context using a previous as the defaults."""

        callback_manager = callback_manager or service_context.callback_manager
        if llm != "default":
            llm = resolve_llm(llm)

        # NOTE: the embed_model isn't used in all indices
        # default to using the embed model passed from the service context
        if embed_model == "default":
            embed_model = service_context.embed_model
        embed_model = resolve_embed_model(embed_model)
        embed_model.callback_manager = callback_manager


        for transform in service_context.transformations:
            if isinstance(transform, NodeParser):
                node_parser = transform
                break

        if text_splitter is not None and node_parser is not None:
            raise ValueError("Cannot specify both text_splitter and node_parser")


        transformations = transformations or service_context.transformations

        return ServiceContext(
            llm=llm,
            embed_model=embed_model,
            transformations=transformations,
            callback_manager=callback_manager,
        )


    def to_dict(self) -> dict:
        """Convert service context to dict."""
        llm_dict = self.llm.to_dict()

        embed_model_dict = self.embed_model.to_dict()

        tranform_list_dict = [x.to_dict() for x in self.transformations]

        return ServiceContextData(
            llm=llm_dict,
            embed_model=embed_model_dict,
            transformations=tranform_list_dict,
        ).model_dump()