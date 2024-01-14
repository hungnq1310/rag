"""IndexStructType class."""

from enum import Enum


class IndexStructType(str, Enum):
    """Index struct type. Identifier for a "type" of index.

    Attributes:
        LIST ("list"): Summary index. See :ref:`Ref-Indices-List` for summary indices.
        KEYWORD_TABLE ("keyword_table"): Keyword table index. See
            :ref:`Ref-Indices-Table`
            for keyword table indices.
        DICT ("dict"): Faiss Vector Store Index. See
            :ref:`Ref-Indices-VectorStore`
            for more information on the faiss vector store index.
        SIMPLE_DICT ("simple_dict"): Simple Vector Store Index. See
            :ref:`Ref-Indices-VectorStore`
            for more information on the simple vector store index.
        MILVUS ("milvus"): Milvus Vector Store Index.
            See :ref:`Ref-Indices-VectorStore`
            for more information on the Milvus vector store index.
        DOCUMENT_SUMMARY ("document_summary"): Document Summary Index.
            See :ref:`Ref-Indices-Document-Summary` for Summary Indices.

    """

    # TODO: refactor so these are properties on the base class

    NODE = "node"
    LIST = "list"
    KEYWORD_TABLE = "keyword_table"

    # faiss
    DICT = "dict"
    # simple
    SIMPLE_DICT = "simple_dict"
    MILVUS = "milvus"
    VECTOR_STORE = "vector_store"

    # EMPTY
    EMPTY = "empty"