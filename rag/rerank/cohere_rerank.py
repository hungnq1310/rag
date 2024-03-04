import os
from typing import Any, List, Optional
import json

from rag.bridge.pydantic import Field, PrivateAttr
from rag.callbacks import CBEventType, EventPayload
from rag.node.base_node import NodeWithScore
from rag.retrievers.types import QueryBundle

from .base import BaseNodePostprocessor


class CohereRerank(BaseNodePostprocessor):
    model: str = Field(description="Cohere model name.")
    top_n: int = Field(description="Top N nodes to return.")

    _client: Any = PrivateAttr()

    def __init__(
        self,
        top_n: int = 2,
        model: str = "rerank-english-v2.0",
        api_key: Optional[str] = None,
    ):
        try:
            api_key = api_key or os.environ["COHERE_API_KEY"]
        except IndexError:
            raise ValueError(
                "Must pass in cohere api key or "
                "specify via COHERE_API_KEY environment variable "
            )
        try:
            from cohere import Client
        except ImportError:
            raise ImportError(
                "Cannot import cohere package, please `pip install cohere`."
            )

        super().__init__(top_n=top_n, model=model)
        self._client = Client(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "CohereRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            texts = [node.node.get_content() for node in nodes]
            results = self._client.rerank(
                model=self.model,
                top_n=self.top_n,
                query=query_bundle.query_str,
                documents=texts,
            )

            new_order = []
            new_nodes = []

            for result in results:
                new_node_with_score = NodeWithScore(
                    node=nodes[result.index].node, score=result.relevance_score
                )
                new_nodes.append(new_node_with_score)
                new_order.append(result.index)
            event.on_end(payload={EventPayload.NODES: new_nodes})

        # EVAL: Saving for evaluatate
        data = {
            "query": query_bundle.query_str,
            "old_order": [i for i in len(nodes)],
            "doc": [node.node.get_content() for node in nodes],
            "new_order": new_order,
            "new_doc": [node.node.get_content() for node in new_nodes],
        }
        with open("artifacts/evaluate_rerank.json", "a") as f:
            f.write(json.dumps(data) + "\n")

        return new_nodes