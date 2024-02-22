"""Node postprocessor."""

import logging
from typing import Dict, List, Optional, cast

from rag.bridge.pydantic import Field
from rag.node.base_node import NodeWithScore
from rag.retrievers.types import QueryBundle

from .base import BaseNodePostprocessor

logger = logging.getLogger(__name__)


class KeywordNodePostprocessor(BaseNodePostprocessor):
    """Keyword-based Node processor."""

    required_keywords: List[str] = Field(default_factory=list)
    exclude_keywords: List[str] = Field(default_factory=list)
    lang: str = Field(default="en")

    @classmethod
    def class_name(cls) -> str:
        return "KeywordNodePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "Spacy is not installed, please install it with `pip install spacy`."
            )
        from spacy.matcher import PhraseMatcher

        nlp = spacy.blank(self.lang)
        required_matcher = PhraseMatcher(nlp.vocab)
        exclude_matcher = PhraseMatcher(nlp.vocab)
        required_matcher.add("RequiredKeywords", list(nlp.pipe(self.required_keywords)))
        exclude_matcher.add("ExcludeKeywords", list(nlp.pipe(self.exclude_keywords)))

        new_nodes = []
        for node_with_score in nodes:
            node = node_with_score.node
            doc = nlp(node.get_content())
            if self.required_keywords and not required_matcher(doc):
                continue
            if self.exclude_keywords and exclude_matcher(doc):
                continue
            new_nodes.append(node_with_score)

        return new_nodes


class SimilarityPostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor."""

    similarity_cutoff: float = Field(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "SimilarityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        sim_cutoff_exists = self.similarity_cutoff is not None

        new_nodes = []
        for node in nodes:
            should_use_node = True
            if sim_cutoff_exists:
                similarity = node.score
                if similarity is None:
                    should_use_node = False
                elif cast(float, similarity) < cast(float, self.similarity_cutoff):
                    should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes