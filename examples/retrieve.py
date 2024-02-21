from pathlib import Path
import time

from rag.pipeline.retriever_pipeline import RetrieverPipeline
from rag.config.configuration import ConfigurationManager

# tạo database 
# input data

from rag.reader.directory_reader import DirectoryReader
import glob

startTime_load = int(round(time.time() * 1000))

pdf_files = glob.glob("/home/hungnq/hungnq_2/rag_pdf/rag_pdf_services/data/*.pdf")
print("pdf_files: ", pdf_files)
reader = DirectoryReader(
    input_files=pdf_files
)
print("reader: ", reader.__dict__)
pdf_documents = reader.load_data()
if pdf_documents:
    print("Load pdf success")

# ------------------

# config
# params
# service context
# synthesizer
# node processor

manager = ConfigurationManager(
    config_filepath=Path("/home/hungnq/hungnq_2/rag_pdf/rag_pdf_services/configs/config.yaml"),
)
node_parser_config = manager.get_splitter_config()
milvus_config = manager.get_milvus_config()
embed_config = manager.get_embed_config()
index_retriver_config = manager.get_index_retriever_config()
response_config = manager.get_response_config()



# # llm
# llm = HuggingFaceLLM(
#     context_window=llm_params.context_window,
#     max_new_tokens=llm_params.max_new_tokens,
#     tokenizer_name=llm_params.tokenizer_name,
#     model_name=llm_params.model_name,
#     device_map=llm_params.device_map,
#     generate_kwargs={
#         "top_p": llm_params.top_p,
#         "top_k": llm_params.top_k,
#         "temperature": llm_params.temperature,
#         "length_penalty": llm_params.length_penalty,
#         "repetition_penalty": llm_params.repetition_penalty,
#         "num_beams": llm_params.num_beams,
#         "do_sample" : llm_params.do_sample,
#         "pelnaty_alpha": llm_params.pelnaty_alpha,
#         "use_cache": llm_params.use_cache,
#         "num_return_sequences": llm_params.num_return_sequences,
#         "pad_token_id": llm_params.pad_token_id,
#         "bos_token_id": llm_params.bos_token_id,
#         "eos_token_id": llm_params.eos_token_id
#     }
# )


# cần 2 params and config: của milvus và index -> done

retriever_pipeline = RetrieverPipeline(
    splitter_config=node_parser_config,
    milvus_config=milvus_config,
    index_retriver_config=index_retriver_config,
    embed_config=embed_config,
    response_config=response_config,
)

endTime_load = int(round(time.time() * 1000))
print(f"Time for load pipeline: {endTime_load - startTime_load} ms")

while True:
    # Main
    query_user = input("Input: ")
    startTime = int(round(time.time() * 1000))
    nodes = retriever_pipeline.main(
        query=query_user,
        documents=pdf_documents,
    )
    for node in nodes:
        text = node.get_content().strip()
        metadata_info = node.node.get_metadata_str()
        score = node.score
        print( f"Node ID:{node.node_id}\nMETADATA\n:{metadata_info}\nText:\n'''{text}'''\nScore:{score}")
    endTime = int(round(time.time() * 1000))
    print(f"Time for retriever_pipeline: {endTime - startTime} ms")
