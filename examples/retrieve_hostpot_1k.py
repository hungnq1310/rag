from rag.engine.retriever_engine import RetrieverEngine
from rag.node_parser.text.sentence import SentenceSplitter
from rag.callbacks import CallbackManager
from rag.core.storage_context import StorageContext
from rag.core.service_context import ServiceContext
from rag.indices.vector_store import VectorStoreIndex
from rag.vector_stores.milvus import MilvusVectorStore
from rag.retrievers.dense.vector_retriver import VectorIndexRetriever
from rag.rerank.sbert_rerank import SentenceTransformerRerank
from rag.rerank.simple import DeltaSimilarityPostprocessor
from rag.embeddings.sbert import SBertEmbedding


from datasets import load_dataset

from rag.node.base_node import MetadataMode
from transformers import AutoTokenizer
from dataclasses import asdict

from rag.config.schema import (
    NodeParserConfig,
    RetrieverConfig,
    EmbeddingConfig,
    MilvusConfig,
    RerankConfig,
)

splitter_config = NodeParserConfig(
    model_name_tokenizer="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # fpr sentence splitter
    splitter_mode="sentence",
    separator=" ",
    chunk_size=256,
    chunk_overlap=round(0.2 * 256),
    paragraph_separator="\n\n\n",
    secondary_chunking_regex="[^.。？！]+[.。？！]?",
    backup_separators=None,
)

retriever_config = RetrieverConfig(
    retriever_mode="or",
    similarity_top_k=10,
    sparse_top_k=20,
    alpha=None,
    list_query_mode=None,
    keyword_table_mode="simple",
    max_keywords_per_chunk=None,
    max_keywords_per_query=10,
    num_chunks_per_query=10,
    vector_store_query_mode="default",
    use_async=False,
    show_progress=True,
    choice_batch_size=1,
    hybrid_mode="dense_sparse",
)

embed_config = EmbeddingConfig(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    tokenizer_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    pooling="cls",
    max_length=512,
    normalize=True,
    embedding_batch_size=16,
    cache_folder=None,
    trust_remote_code=True,
    use_async=False,
    show_progress=True,
    token="",
    device="cuda:0",
)

milvus_config = MilvusConfig(
    vectorstore_name="milvus",
    host="localhost",
    port=19530,
    address=None,
    uri=None,
    user=None,
    collection_name="wiki_tb",
    insert_batch_size=2048,
    embedding_dim=384,
    embedding_field="embedding",
    primary_field="id",
    text_field="text",
    consistency_level=None,
    overwrite=False,
    search_params={
        "metric_type": "COSINE",
        "offset": 0,
        "ignore_growing": False,
        "params": {"nprobe": 10},
    },
    index_params={
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    },
)

rerank_config = RerankConfig(
    top_n=5,
    model="rerank-multilingual-v2.0",
    api_key="xxx",
    modes=["delta_cutoff", "sbert"],
    use_async=False,
    show_progress=True,
    model_name="BAAI/bge-reranker-large",
    token=None,
    device="cuda:0",
    tokenizer_name="None",
    max_length=512,
    keep_retrieval_score=True,
    choice_batch_size=1,
    delta_similarity_cutoff=0.7,
)

# connect milvus

#callback manager
callback_manager = CallbackManager()

# node parser
tokenizer = AutoTokenizer.from_pretrained(splitter_config.model_name_tokenizer)
node_parser = SentenceSplitter(
    separator= splitter_config.separator,
    chunk_size= splitter_config.chunk_size,
    chunk_overlap= splitter_config.chunk_overlap,
    tokenizer= tokenizer.encode,
    paragraph_separator= splitter_config.paragraph_separator,
    secondary_chunking_regex= splitter_config.secondary_chunking_regex,
    callback_manager= callback_manager,
)

emb_model = SBertEmbedding(
    model_name_or_path=embed_config.model_name,
    max_length= embed_config.max_length,
    embed_batch_size= embed_config.embedding_batch_size,
    cache_folder= embed_config.cache_folder,
)

service_context = ServiceContext(
    embed_model=emb_model,
    node_parser=node_parser,
    callback_manager=callback_manager,
)

milvus_vector_store = MilvusVectorStore(**asdict(milvus_config))

# construct index and customize storage context
storage_context = StorageContext.from_defaults(
    vector_store= milvus_vector_store
)

index = VectorStoreIndex(
    storage_context=storage_context,
    service_context=service_context,
    store_nodes_override=True,
    use_async=retriever_config.use_async,
    show_progress=retriever_config.show_progress,
)

# TODO: build retrievers
dense_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=retriever_config.similarity_top_k,
    sparse_top_k=retriever_config.sparse_top_k,
    alpha=retriever_config.alpha,
    vector_store_query_mode=retriever_config.vector_store_query_mode,
)

#TODO: add keyword search maybe more reasonable


model_rerank = SentenceTransformerRerank(
    top_n=rerank_config.top_n,
    model_name=rerank_config.model_name,
    device=rerank_config.device,
    keep_retrieval_score=rerank_config.keep_retrieval_score,
)
mean_delta_rerank = DeltaSimilarityPostprocessor(
    delta_similarity_cutoff=rerank_config.delta_similarity_cutoff
)

# TODO: assemble query engine
query_engine = RetrieverEngine(
    retriever=dense_retriever, node_postprocessors=[model_rerank]
)

# load data from hub

dataset = load_dataset("hero-nq1310/hostpot_1k_test", token="xxx")
data_hostpot1k = dataset["train"]
print(f"Data loaded: {len(data_hostpot1k)} samples")

def retrieve_chunk(entry): 
    questions = entry['question']
    retrieved_chunks = []
    for question in questions:
        retrieved_nodes = query_engine.retrieve(
            question
        )
        print(retrieved_nodes)
        retrieved_chunks.append(
            [node.text for node in retrieved_nodes]
        )
    entry['retrieved_contexts'] = retrieved_chunks
    return entry

for e_data in data_hostpot1k:
    question = e_data['question']
    retrieved_chunks = []

    retrieved_nodes = query_engine.retrieve(
        question
    )
    print(retrieved_nodes)
    retrieved_chunks.append(
        [node.text for node in retrieved_nodes]
    )
    e_data['retrieved_contexts'] = retrieved_chunks
    
data_hostpot1k.save_to_disk("data/evaluate/")