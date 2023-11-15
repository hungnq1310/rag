"""
Metadata extractors for nodes. Applied as a post processor to node parsing.
Currently, only `TextNode` is supported.

Supported metadata:
Document-level:
    - `TitleExtractor`: Document title, possible inferred across multiple nodes

Unimplemented (contributions welcome):
Subsection:
    - Position of node in subsection hierarchy (and associated subtitles)
    - Hierarchically organized summary

The prompts used to generate the metadata are specifically aimed to help
disambiguate the document or subsection from other similar documents or subsections.
(similar with contrastive learning)
"""
from abc import abstractmethod
from copy import deepcopy
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Sequence, cast

from rag.bridge.pydantic import Field, PrivateAttr
from llama_index.llm_predictor.base import BaseLLMPredictor, LLMPredictor
from llama_index.llms.base import LLM
from rag.entity.node_parser import BaseExtractor
from rag.entity.prompts import PromptTemplate
from rag.entity.schema import BaseNode, MetadataMode, TextNode
from rag.entity.types import BasePydanticProgram
from rag.utils import get_tqdm_iterable


class MetadataFeatureExtractor(BaseExtractor):
    is_text_node_only: bool = True
    show_progress: bool = True
    metadata_mode: MetadataMode = MetadataMode.ALL

    @abstractmethod
    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extracts metadata for a sequence of nodes, returning a list of
        metadata dictionaries corresponding to each node.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """


DEFAULT_NODE_TEXT_TEMPLATE = """\
[Excerpt from document]\n{metadata_str}\n\
Excerpt:\n-----\n{content}\n-----\n"""


class MetadataExtractor(BaseExtractor):
    """Metadata extractor."""

    extractors: Sequence[MetadataFeatureExtractor] = Field(
        default_factory=list,
        description="Metadta feature extractors to apply to each node.",
    )
    node_text_template: str = Field(
        default=DEFAULT_NODE_TEXT_TEMPLATE,
        description="Template to represent how node text is mixed with metadata text.",
    )
    disable_template_rewrite: bool = Field(
        default=False, description="Disable the node template rewrite."
    )

    in_place: bool = Field(
        default=True, description="Whether to process nodes in place."
    )

    @classmethod
    def class_name(cls) -> str:
        return "MetadataExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        """Extract metadata from a document.

        Args:
            nodes (Sequence[BaseNode]): nodes to extract metadata from

        """
        metadata_list: List[Dict] = [{} for _ in nodes]
        for extractor in self.extractors:
            cur_metadata_list = extractor.extract(nodes)
            for i, metadata in enumerate(metadata_list):
                metadata.update(cur_metadata_list[i])

        return metadata_list

    def process_nodes(
        self,
        nodes: List[BaseNode],
        excluded_embed_metadata_keys: Optional[List[str]] = None,
        excluded_llm_metadata_keys: Optional[List[str]] = None,
    ) -> List[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
            excluded_embed_metadata_keys (Optional[List[str]]):
                keys to exclude from embed metadata
            excluded_llm_metadata_keys (Optional[List[str]]):
                keys to exclude from llm metadata
        """
        if self.in_place:
            new_nodes = nodes
        else:
            new_nodes = [deepcopy(node) for node in nodes]
        for extractor in self.extractors:
            cur_metadata_list = extractor.extract(new_nodes)
            for idx, node in enumerate(new_nodes):
                node.metadata.update(cur_metadata_list[idx])

        for idx, node in enumerate(new_nodes):
            if excluded_embed_metadata_keys is not None:
                node.excluded_embed_metadata_keys.extend(excluded_embed_metadata_keys)
            if excluded_llm_metadata_keys is not None:
                node.excluded_llm_metadata_keys.extend(excluded_llm_metadata_keys)
            if not self.disable_template_rewrite:
                if isinstance(node, TextNode):
                    cast(TextNode, node).text_template = self.node_text_template
        return new_nodes
    

DEFAULT_TITLE_NODE_TEMPLATE = """\
Context: {context_str}. Give a title that summarizes all of \
the unique entities, titles or themes found in the context. Title: """


DEFAULT_TITLE_COMBINE_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
what is the comprehensive title for this document? Title: """
    
    
class TitleExtractor(MetadataFeatureExtractor):
    """Title extractor. Useful for long documents. Extracts `document_title`
    metadata field.

    Args:
        llm_predictor (Optional[BaseLLMPredictor]): LLM predictor
        nodes (int): number of nodes from front to use for title extraction
        node_template (str): template for node-level title clues extraction
        combine_template (str): template for combining node-level clues into
            a document-level title
    """

    is_text_node_only: bool = False  # can work for mixture of text and non-text nodes
    llm_predictor: BaseLLMPredictor = Field(
        description="The LLMPredictor to use for generation."
    )
    nodes: int = Field(
        default=5, description="The number of nodes to extract titles from."
    )
    node_template: str = Field(
        default=DEFAULT_TITLE_NODE_TEMPLATE,
        description="The prompt template to extract titles with.",
    )
    combine_template: str = Field(
        default=DEFAULT_TITLE_COMBINE_TEMPLATE,
        description="The prompt template to merge titles with.",
    )

    def __init__(
        self,
        llm: Optional[LLM] = None,
        # TODO: llm_predictor arg is deprecated
        llm_predictor: Optional[BaseLLMPredictor] = None,
        nodes: int = 5,
        node_template: str = DEFAULT_TITLE_NODE_TEMPLATE,
        combine_template: str = DEFAULT_TITLE_COMBINE_TEMPLATE,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if nodes < 1:
            raise ValueError("num_nodes must be >= 1")

        if llm is not None:
            llm_predictor = LLMPredictor(llm=llm)
        elif llm_predictor is None and llm is None:
            llm_predictor = LLMPredictor()

        super().__init__(
            llm_predictor=llm_predictor,
            nodes=nodes,
            node_template=node_template,
            combine_template=combine_template,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TitleExtractor"

    def extract(self, nodes: Sequence[BaseNode]) -> List[Dict]:
        nodes_to_extract_title: List[BaseNode] = []
        for node in nodes:
            if len(nodes_to_extract_title) >= self.nodes:
                break
            if self.is_text_node_only and not isinstance(node, TextNode):
                continue
            nodes_to_extract_title.append(node)

        if len(nodes_to_extract_title) == 0:
            # Could not extract title
            return []

        title_candidates = [
            self.llm_predictor.predict(
                PromptTemplate(template=self.node_template),
                context_str=cast(TextNode, node).text,
            )
            for node in nodes_to_extract_title
        ]
        if len(nodes_to_extract_title) > 1:
            titles = reduce(
                lambda x, y: x + "," + y, title_candidates[1:], title_candidates[0]
            )

            title = self.llm_predictor.predict(
                PromptTemplate(template=self.combine_template),
                context_str=titles,
            )
        else:
            title = title_candidates[
                0
            ]  # if single node, just use the title from that node

        return [{"document_title": title.strip(' \t\n\r"')} for _ in nodes]