"""Index registry."""

from typing import Dict, Type

from rag.indices.data_struct import (
    EmptyIndexStruct,
    IndexDict,
    IndexList,
    IndexStruct,
    KeywordTable,
)
from rag.indices.types import IndexStructType

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[IndexStruct]] = {
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.EMPTY: EmptyIndexStruct,
}
