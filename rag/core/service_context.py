import logging
from dataclasses import dataclass
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
from rag.llm.base import LLM, LLMType, LLMMetadata
from rag.node_parser.base import TextSplitter
from .prompt_helper import PromptHelper

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


def _get_default_prompt_helper(
    llm_metadata: LLMMetadata = None,
    context_window: Optional[int] = None,
    num_output: Optional[int] = None,
) -> PromptHelper:
    """Get default prompt helper."""
    if context_window is not None:
        llm_metadata.context_window = context_window
    if num_output is not None:
        llm_metadata.num_output = num_output
    return PromptHelper.from_llm_metadata(llm_metadata=llm_metadata)


class ServiceContextData(BaseModel):
    llm: dict
    prompt_helper: dict
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
    llm: LLM = None
    prompt_helper: PromptHelper = None
    embed_model: BaseEmbedding = None
    transformations: List[TransformComponent] = None
    callback_manager: CallbackManager = None

    def __init__(self,
        llm: Optional[LLMType] = None,
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[EmbedType] = None,
        node_parser: Optional[SentenceSplitter] = None,
        text_splitter: Optional[TextSplitter] = None,
        transformations: Optional[List[TransformComponent]] = None,
        callback_manager: Optional[CallbackManager] = None,
        # node parser kwargs
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        # prompt helper kwargs
        context_window: Optional[int] = None,
        num_output: Optional[int] = None,
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

        self.prompt_helper = prompt_helper or _get_default_prompt_helper(
            llm_metadata=self.llm.metadata,
            context_window=context_window,
            num_output=num_output,
        )

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
        prompt_helper: Optional[PromptHelper] = None,
        embed_model: Optional[EmbedType] = "default",
        node_parser: Optional[SentenceSplitter] = None,
        text_splitter: Optional[TextSplitter] = None,
        transformations: Optional[List[TransformComponent]] = None,
        callback_manager: Optional[CallbackManager] = None,
        # node parser kwargs
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        # prompt helper kwargs
        context_window: Optional[int] = None,
        num_output: Optional[int] = None,
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

        prompt_helper = prompt_helper or service_context.prompt_helper
        if context_window is not None or num_output is not None:
            prompt_helper = _get_default_prompt_helper(
                llm_metadata=llm.metadata,
                context_window=context_window,
                num_output=num_output,
            )

        transformations = transformations or []
        node_parser_found = False
        for transform in service_context.transformations:
            if isinstance(transform, SentenceSplitter):
                node_parser_found = True
                node_parser = transform
                break

        if text_splitter is not None and node_parser is not None:
            raise ValueError("Cannot specify both text_splitter and node_parser")

        if not node_parser_found:
            node_parser = (
                text_splitter  # text splitter extends node parser
                or node_parser
                or _get_default_node_parser(
                    chunk_size=chunk_size or DEFAULT_CHUNK_SIZE,
                    chunk_overlap=chunk_overlap or SENTENCE_CHUNK_OVERLAP,
                    callback_manager=callback_manager,
                )
            )

        transformations = transformations or service_context.transformations

        return ServiceContext(
            llm=llm,
            embed_model=embed_model,
            prompt_helper=prompt_helper,
            transformations=transformations,
            callback_manager=callback_manager,
        )


    @property
    def node_parser(self) -> SentenceSplitter:
        """Get the node parser."""
        for transform in self.transformations:
            if isinstance(transform, SentenceSplitter):
                return transform
        raise ValueError("No node parser found.")

    def to_dict(self) -> dict:
        """Convert service context to dict."""
        llm_dict = self.llm.to_dict()

        embed_model_dict = self.embed_model.to_dict()

        prompt_helper_dict = self.prompt_helper.to_dict()

        tranform_list_dict = [x.to_dict() for x in self.transformations]

        return ServiceContextData(
            llm=llm_dict,
            prompt_helper=prompt_helper_dict,
            embed_model=embed_model_dict,
            transformations=tranform_list_dict,
        ).model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> "ServiceContext":
        from llama_index.embeddings.loading import load_embed_model
        from llama_index.extractors.loading import load_extractor
        from llama_index.llms.loading import load_llm
        from llama_index.node_parser.loading import load_parser

        service_context_data = ServiceContextData.parse_obj(data)

        llm = load_llm(service_context_data.llm)

        embed_model = load_embed_model(service_context_data.embed_model)

        prompt_helper = PromptHelper.from_dict(service_context_data.prompt_helper)

        transformations: List[TransformComponent] = []
        for transform in service_context_data.transformations:
            try:
                transformations.append(load_parser(transform))
            except ValueError:
                transformations.append(load_extractor(transform))

        return cls.from_defaults(
            llm=llm,
            prompt_helper=prompt_helper,
            embed_model=embed_model,
            transformations=transformations,
        )
