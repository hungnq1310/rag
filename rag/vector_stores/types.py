"""Vector store index types."""
from enum import Enum
from dataclasses import dataclass
from typing import (
    Union,
    Optional,
    Sequence,
    List,
    Dict
)

from rag.node.base_node import BaseNode
from rag.bridge.pydantic import BaseModel, StrictInt, StrictFloat, StrictStr


class ExactMatchFilter(BaseModel):
    """Exact match metadata filter for vector stores.

    Value uses Strict* types, as int, float and str are compatible types and were all
    converted to string before.

    See: https://docs.pydantic.dev/latest/usage/types/#strict-types
    """

    key: str
    value: Union[StrictInt, StrictFloat, StrictStr]


class MetadataFilters(BaseModel):
    """Metadata filters for vector stores.

    Currently only supports exact match filters.
    TODO: support more advanced expressions.
    """

    filters: List[ExactMatchFilter]

    @classmethod
    def from_dict(cls, filter_dict: Dict) -> "MetadataFilters":
        """Create MetadataFilters from json."""
        filters = []
        for k, v in filter_dict.items():
            filter = ExactMatchFilter(key=k, value=v)
            filters.append(filter)
        return cls(filters=filters)

class VectorStoreQueryMode:
    """Vector store query mode."""

    DEFAULT = "default"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    TEXT_SEARCH = "text_search"

    # fit learners
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"

    # maximum marginal relevance
    MMR = "mmr"

@dataclass
class VectorStoreQueryResult:
    """Vector store query result."""

    nodes: Optional[Sequence[BaseNode]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None


@dataclass
class VectorStoreQuery:
    """Vector store query."""

    query_embedding: Optional[List[float]] = None
    similarity_top_k: int = 1
    doc_ids: Optional[List[str]] = None
    node_ids: Optional[List[str]] = None
    query_str: Optional[str] = None
    output_fields: Optional[List[str]] = None
    embedding_field: Optional[str] = None

    mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT

    # NOTE: only for hybrid search (0 for bm25, 1 for vector search)
    alpha: Optional[float] = None

    # metadata filters
    filters: Optional[MetadataFilters] = None

    # only for mmr
    mmr_threshold: Optional[float] = None

    # NOTE: currently only used by postgres hybrid search
    sparse_top_k: Optional[int] = None

class FilterOperator(str, Enum):
    """Vector store filter operator."""

    # TODO add more operators
    EQ = "=="  # default operator (string, int, float)
    GT = ">"  # greater than (int, float)
    LT = "<"  # less than (int, float)
    NE = "!="  # not equal to (string, int, float)
    GTE = ">="  # greater than or equal to (int, float)
    LTE = "<="  # less than or equal to (int, float)
    IN = "in"  # In array (string or number)
    NIN = "nin"  # Not in array (string or number)
    TEXT_MATCH = "text_match"  # full text match (allows you to search for a specific substring, token or phrase within the text field)