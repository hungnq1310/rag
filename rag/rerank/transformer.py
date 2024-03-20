from typing import Any, List, Optional

from rag.bridge.pydantic import Field, PrivateAttr
from rag.callbacks import CBEventType, EventPayload
from rag.node.base_node import MetadataMode, NodeWithScore
from rag.retrievers.types import QueryBundle
from rag.rag_utils import infer_torch_device

from .base import BaseNodePostprocessor

DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH = 512


class TransformerRerank(BaseNodePostprocessor):
    # Attribute
    model_name_or_path: str = Field(description="transformer model name.")
    tokenizer_name: str = Field(description="transformer tokenizer name.")
    top_n: int = Field(description="Number of nodes to return sorted by score.")
    device: str = Field(
        default="cpu",
        description="Device to use for sentence transformer.",
    )
    keep_retrieval_score: bool = Field(
        default=False,
        description="Whether to keep the retrieval score in metadata.",
    )
    token: str = Field(default=None, description="Token to use for huggingface transformer.",
    )

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        token: Optional[str] = None,
        top_n: int = 2,
        device: Optional[str] = None,
        keep_retrieval_score: Optional[bool] = False,
    ):
        try:
            from transformers import AutoTokenizer, AutoModel  # pants: no-infer-dep
        except ImportError:
            raise ImportError(
                "Cannot import sentence-transformers or torch package,",
                "please `pip install torch sentence-transformers`",
            )
        device = infer_torch_device() if device is None else device

        # load model
        if model_name_or_path:
            model = AutoModel.from_pretrained(
                model_name_or_path, token=token
            )
            self._model = model.to(device)
        else:
            self._model = None
            
        # load tokenizer
        if tokenizer_name is None:
            if model_name_or_path is None:
                raise ValueError('model_name_or_path and tokenizer is None')
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, token=token
            )
        super().__init__(
            model_name_or_path=model_name_or_path,
            tokenizer_name=tokenizer_name,
            token=token,
            top_n=top_n,
            device=device,
            keep_retrieval_score=keep_retrieval_score,
        )

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        import torch

        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        # model input
        query_and_nodes = [
            (
                query_bundle.query_str,
                node.node.get_content(metadata_mode=MetadataMode.EMBED),
            )
            for node in nodes
        ]
        model_inputs = self._tokenizer(
            query_and_nodes, # pairs = [['query', 'doc1'], ['query', 'doc2']]
            padding=True, truncation=True,
            return_tensors='pt'
        ).to("cuda:0")

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model_name_or_path,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            
            with torch.no_grad():
                outputs = self._model(**model_inputs, return_dict=True).logits.view(-1, ).float()
            scores = outputs.cpu().detach().numpy()
            
            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores):
                if self.keep_retrieval_score:
                    # keep the retrieval score in metadata
                    node.node.metadata["retrieval_score"] = node.score
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
                : self.top_n
            ]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes