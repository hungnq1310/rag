"""Data structures.

Nodes are decoupled from the indices.

"""

import uuid
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from dataclasses_json import DataClassJsonMixin

from .types import IndexStructType
from rag.node.base_node import BaseNode, TextNode

# TODO: legacy backport of old Node class
Node = TextNode


@dataclass
class IndexStruct(DataClassJsonMixin):
    """A base data struct for a LlamaIndex."""

    index_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    summary: Optional[str] = None

    def get_summary(self) -> str:
        """Get text summary."""
        if self.summary is None:
            raise ValueError("summary field of the index_struct not set.")
        return self.summary

    @classmethod
    @abstractmethod
    def get_type(cls) -> IndexStructType:
        """Get index struct type."""


@dataclass
class KeywordTable(IndexStruct):
    """A table of keywords mapping keywords to text chunks."""

    table: Dict[str, Set[str]] = field(default_factory=dict)

    def add_node(self, keywords: List[str], node: BaseNode) -> None:
        """Add text to table."""
        for keyword in keywords:
            if keyword not in self.table:
                self.table[keyword] = set()
            self.table[keyword].add(node.node_id)

    @property
    def node_ids(self) -> Set[str]:
        """Get all node ids."""
        return set.union(*self.table.values())

    @property
    def keywords(self) -> Set[str]:
        """Get all keywords in the table."""
        return set(self.table.keys())

    @property
    def size(self) -> int:
        """Get the size of the table."""
        return len(self.table)

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.KEYWORD_TABLE


@dataclass
class IndexList(IndexStruct):
    """A list of documents."""

    nodes: List[str] = field(default_factory=list)

    def add_node(self, node: BaseNode) -> None:
        """Add text to table, return current position in list."""
        # don't worry about child indices for now, nodes are all in order
        self.nodes.append(node.node_id)

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.LIST


@dataclass
class IndexDict(IndexStruct):
    """A simple dictionary of documents."""

    # TODO: slightly deprecated, should likely be a list or set now
    # mapping from vector store id to node doc_id
    nodes_dict: Dict[str, str] = field(default_factory=dict)

    # TODO: deprecated, not used
    # mapping from node doc_id to vector store id
    doc_id_dict: Dict[str, List[str]] = field(default_factory=dict)

    # TODO: deprecated, not used
    # this should be empty for all other indices
    embeddings_dict: Dict[str, List[float]] = field(default_factory=dict)

    def add_node(
        self,
        node: BaseNode,
        text_id: Optional[str] = None,
    ) -> str:
        """Add text to table, return current position in list."""
        # # don't worry about child indices for now, nodes are all in order
        # self.nodes_dict[int_id] = node
        vector_id = text_id if text_id is not None else node.node_id
        self.nodes_dict[vector_id] = node.node_id

        return vector_id

    def delete(self, doc_id: str) -> None:
        """Delete a Node."""
        del self.nodes_dict[doc_id]

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.VECTOR_STORE


@dataclass
class EmptyIndexStruct(IndexStruct):
    """Empty index."""

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.EMPTY