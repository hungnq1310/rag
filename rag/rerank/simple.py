"""Node postprocessor."""

import logging
from typing import List, Optional, cast
import numpy as np

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


class DeltaSimilarityPostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor."""

    delta_similarity_cutoff: float = Field(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "DeltaSimilarityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        sim_cutoff_exists = self.delta_similarity_cutoff is not None

        # get best score of first node
        best_score: float = 0.0
        if nodes:
            best_score = max([cast(float, node.score) for node in nodes])
        
        new_nodes = []
        for node in nodes:
            should_use_node = True
            if sim_cutoff_exists:
                similarity = cast(float, node.score)
                if similarity is None:
                    should_use_node = False
                elif best_score - similarity > self.delta_similarity_cutoff:
                    should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes
    
class MeanDeltaSimilarityPostprocessor(BaseNodePostprocessor):
    """Similarity-based Node processor."""

    delta_similarity_cutoff: float = Field(default=None)

    @classmethod
    def class_name(cls) -> str:
        return "MeanDeltaSimilarityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        sim_cutoff_exists = self.delta_similarity_cutoff is not None
        
        # get the score of all retrieved node
        scores = np.array([cast(float, node.score) for node in nodes])

        # verify distribution
        if scores.min() < 0 or scores.max() > 1:
            # normalize the scores
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min())
            print("Normalizing scores: ", normalized_scores)
        else:
            normalized_scores = scores

        # get mean score of all retrieved node
        mean_score: float = 0.0
        if nodes:
            mean_score = sum(normalized_scores) / len(normalized_scores)
        
        new_nodes = []
        for idx, similarity in enumerate(normalized_scores):
            should_use_node = True
            if sim_cutoff_exists:
                if similarity is None:
                    should_use_node = False
                # compute gap score
                gap_score = -(similarity - mean_score) if similarity < mean_score else similarity - mean_score

                if gap_score > self.delta_similarity_cutoff:
                    should_use_node = False

            if should_use_node:
                new_nodes.append(nodes[idx])

        return new_nodes
    

class MeanSimilarityPostprocessor(BaseNodePostprocessor):

    @classmethod
    def class_name(cls) -> str:
        return "MeanSimilarityPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""

        # get mean score of all retrieved node
        mean_score: float = 0.0
        if nodes:
            mean_score = sum([cast(float, node.score) for node in nodes]) / len(nodes)
        
        new_nodes = []
        for node in nodes:
            should_use_node = True
            # get similiarity score
            similarity = node.score
            if similarity is None:
                should_use_node = False
            similarity = cast(float, similarity)
            # compare similarity score with mean score
            if similarity < mean_score:
                should_use_node = False

            if should_use_node:
                new_nodes.append(node)

        return new_nodes