"""Index registry."""

from typing import Dict, Type

from .data_struct import (
    EmptyIndexStruct,
    IndexDict,
    IndexList,
    IndexStruct,
    KeywordTable,
)
from llama_index.data_structs.document_summary import IndexDocumentSummary
from llama_index.data_structs.struct_type import IndexStructType

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[IndexStruct]] = {
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.EMPTY: EmptyIndexStruct,
    IndexStructType.DOCUMENT_SUMMARY: IndexDocumentSummary,
}
