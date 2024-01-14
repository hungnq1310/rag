from transformers import AutoTokenizer, AutoModel

from rag.pipeline import RetrieverPipeline
from rag.config.configuration import ConfigurationManager
from rag.entity.service_context import ServiceContext
from rag.entity.callbacks import CallbackManager
from rag.components.node_parser import SentenceSplitter
from rag.components.llm import HuggingFaceLLM
from rag.components.embeddings import HuggingFaceEmbedding
from rag.components.synthesizer import get_response_synthesizer
from llama_index import PromptHelper

# tạo database 
# input data

from llama_index import SimpleDirectoryReader
import glob

pdf_files = glob.glob("./data/*.pdf")
reader = SimpleDirectoryReader(
    input_files=pdf_files
)
pdf_documents = reader.load_data()


# ------------------

# config
# params
# service context
# synthesizer
# node processor

manager = ConfigurationManager()
node_parser_config = manager.get_node_parser_config()
node_parser_params = manager.get_node_parser_params()
milvus_config = manager.get_milvus_config()
milvus_params = manager.get_milvus_params()
llm_params = manager.get_llm_params()
embed_params = manager.get_embed_params()
index_retriver_params = manager.get_index_retriever_params()
response_params = manager.get_response_params()

#callback manager
callback_manager = CallbackManager()

# node parser
tokenizer = AutoTokenizer.from_pretrained(node_parser_params.model_name_tokenizer)

node_parser = SentenceSplitter(
    separator=node_parser_params.separator,
    chunk_size=node_parser_params.chunk_size,
    chunk_overlap=node_parser_params.chunk_overlap,
    tokenizer=tokenizer.encode,
    paragraph_separator="\n\n\n",
    secondary_chunking_regex=node_parser_params.secondary_chunking_regex
)

# llm
llm = HuggingFaceLLM(
    context_window=llm_params.context_window,
    max_new_tokens=llm_params.max_new_tokens,
    tokenizer_name=llm_params.tokenizer_name,
    model_name=llm_params.model_name,
    device_map=llm_params.device_map,
    generate_kwargs={
        "top_p": llm_params.top_p,
        "top_k": llm_params.top_k,
        "temperature": llm_params.temperature,
        "length_penalty": llm_params.length_penalty,
        "repetition_penalty": llm_params.repetition_penalty,
        "num_beams": llm_params.num_beams,
        "do_sample" : llm_params.do_sample,
        "pelnaty_alpha": llm_params.pelnaty_alpha,
        "use_cache": llm_params.use_cache,
        "num_return_sequences": llm_params.num_return_sequences,
        "pad_token_id": llm_params.pad_token_id,
        "bos_token_id": llm_params.bos_token_id,
        "eos_token_id": llm_params.eos_token_id
    }
)

# emb_model
emb_model = HuggingFaceEmbedding(
    model_name=embed_params.model_name,
    tokenizer_name=embed_params.tokenizer_name,
    pooling=embed_params.pooling,
    max_length=embed_params.max_length,
    normalize=embed_params.normalize,
    embedding_batch_size=embed_params.embedding_batch_size,
    cache_folder=embed_params.cache_folder,
    trust_remote_code=embed_params.trust_remote_code,
)

# prompt helper - tong hop 3 cai params cua llm, emb, node parser
prompt_helper = PromptHelper()


service_context = ServiceContext(
    llm=llm,
    prompt_helper=prompt_helper,
    embed_model=emb_model,
    callback_manager=callback_manager
)

response_synthesizer = get_response_synthesizer(
    service_context=service_context,
    callback_manager=callback_manager,
    response_mode=response_params.response_mode,
    verbose=response_params.verbose,
    use_async=response_params.use_async,
    streaming=response_params.streaming,
)

# cần 2 params and config: của milvus và index -> done

retriever_pipeline = RetrieverPipeline(
    milvus_config=milvus_config,
    milvus_params=milvus_params,
    index_params=index_retriver_params,
    service_context=service_context,
    response_synthesizer=response_synthesizer,
    node_postprocessors=None,
)

# Main
query_user = input("Input: ")

retriever_pipeline.main(
    query=query_user,
    documents=pdf_documents,
)
