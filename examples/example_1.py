from pathlib import Path
import time
import os

from rag.pipeline.milvus_retriever import MilvusRetrieverPipeline
from rag.config.configuration import ConfigurationManager

# tạo database 
# input data

from rag.reader.directory_reader import DirectoryReader
import glob

startTime_load = int(round(time.time() * 1000))

pdf_files = glob.glob(os.path.abspath(os.curdir) + "/data/*.pdf")
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


# cần 2 params and config: của milvus và index -> done
retriever_pipeline = MilvusRetrieverPipeline(
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
        print("-" * 20)
        text = node.get_content().strip()
        metadata_info = node.node.get_metadata_str()
        score = node.score
        print( f"Node ID:{node.node_id}\nMETADATA\n:{metadata_info}\nText:\n'''{text}'''\nScore:{score}")
    endTime = int(round(time.time() * 1000))
    print(f"Time for retriever_pipeline: {endTime - startTime} ms")
