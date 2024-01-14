""" For Huggingface Embeddings"""
DEFAULT_HUGGINGFACE_EMBEDDING_MODEL = "BAAI/bge-small-en"
DEFAULT_INSTRUCT_MODEL = "hkunlp/instructor-base"

# Originally pulled from:
# https://github.com/langchain-ai/langchain/blob/v0.0.257/libs/langchain/langchain/embeddings/huggingface.py#L10
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = (
    "Represent the question for retrieving supporting documents: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_EN = (
    "Represent this question for searching relevant passages: "
)
DEFAULT_QUERY_BGE_INSTRUCTION_ZH = "为这个句子生成表示以用于检索相关文章："

BGE_MODELS = (
    "BAAI/bge-small-en",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en",
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-small-zh",
    "BAAI/bge-small-zh-v1.5",
    "BAAI/bge-base-zh",
    "BAAI/bge-base-zh-v1.5",
    "BAAI/bge-large-zh",
    "BAAI/bge-large-zh-v1.5",
)
INSTRUCTOR_MODELS = (
    "hku-nlp/instructor-base",
    "hku-nlp/instructor-large",
    "hku-nlp/instructor-xl",
    "hkunlp/instructor-base",
    "hkunlp/instructor-large",
    "hkunlp/instructor-xl",
)

