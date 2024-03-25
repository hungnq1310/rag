"""General node utils."""


import logging
import uuid
from typing import List, Optional, Protocol, runtime_checkable

from rag.node.base_node import (
    BaseNode,
    Document,
    NodeRelationship,
    TextNode,
)
from rag.callbacks.callback_manager import CallbackManager
from rag.node_parser.text.sentence import (
    DEFAULT_CHUNK_SIZE,
    SENTENCE_CHUNK_OVERLAP,
    SentenceSplitter,
)
from rag.rag_utils.utils import truncate_text

logger = logging.getLogger(__name__)


@runtime_checkable
class IdFuncCallable(Protocol):
    def __call__(self, i: int, doc: BaseNode) -> str:
        ...


def default_id_func(i: int, doc: BaseNode) -> str:
    return str(uuid.uuid4())


def get_default_node_parser(
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


def build_nodes_from_splits(
    text_splits: List[str],
    document: BaseNode,
    ref_doc: Optional[BaseNode] = None,
    id_func: Optional[IdFuncCallable] = None,
) -> List[TextNode]:
    """Build nodes from splits."""
    ref_doc = ref_doc or document
    id_func = id_func or default_id_func
    nodes: List[TextNode] = []
    for i, text_chunk in enumerate(text_splits):
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")

        if isinstance(document, Document):
            node = TextNode(
                id_=id_func(i, document),
                text=text_chunk,
                embedding=document.embedding,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
            )
            nodes.append(node)
        elif isinstance(document, TextNode):
            node = TextNode(
                id_=id_func(i, document),
                text=text_chunk,
                embedding=document.embedding,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
            )
            nodes.append(node)
        else:
            raise ValueError(f"Unknown document type: {type(document)}")

    return nodes