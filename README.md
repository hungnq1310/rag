# rag_pdf_services

# TODO: 
# Setup Environment
```
conda create -n backend python==3.10 -y
conda activate backend
pip install -r requirements.txt
```
# Setup Milvus server if pipeline using milvus

### Requirement
Milvus server run on Docker server, please see these following links first:
1. https://docs.docker.com/engine/install/
2. https://milvus.io/docs/prerequisite-docker.md

### Install Milvus
Download milvus-standalone-docker-compose.yml and save it as docker-compose.yml manually, or with the following command:
```
wget https://github.com/milvus-io/milvus/releases/download/v2.3.2/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
### Start Milvus
In the same directory as the docker-compose.yml file, start up Milvus by running:
```
sudo docker-compose up -d
```

Now check if same directory as the docker-compose.yml file has folder volumes, and run the following command:
```
sudo docker compose ps
```
Connect to Milvus through port:
```
docker port milvus-standalone 19530/tcp
```

# define config
# use pipeline
Supporting pipeline:
1. BM25
2. Decompose using milvus vector
3. Dense retriever using milvus
4. Keyword retriver (extract keyword from user question and compare with docs)

# Run code

Folder `examples/` has some usecases, generally split into three steps:
1. call `DirectoryReader` from `src.reader.dir_reader` and put list of file path to argument `input_files`

2. initialize Hydra config and put into argument `config` of `ConfigurationManager`

3. Call class pipeline in folder `src.pipeline`, put config and run `main()` function. Currently only supporting pipeline for data ingestion.  



# test/eval
# references