import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path

import requests
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.writers import DocumentWriter

from neo4j_haystack import Neo4jDocumentStore

logger = logging.getLogger(__name__)


def fetch_archive_from_http(url: str, output_dir: str):
    if Path(output_dir).is_dir():
        logger.warn(f"'{output_dir}' directory already exists. Skipping data download")
        return

    with requests.get(url, timeout=10, stream=True) as response:
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(output_dir)


# Let's first get some files that we want to use
docs_dir = "data/docs"
fetch_archive_from_http(
    url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt6.zip",
    output_dir=docs_dir,
)

# Make sure you have a running Neo4j database, e.g. with Docker:
# docker run \
#     --restart always \
#     --publish=7474:7474 --publish=7687:7687 \
#     --env NEO4J_AUTH=neo4j/passw0rd \
#     neo4j:5.15.0

document_store = Neo4jDocumentStore(
    url="bolt://localhost:7687",
    username="neo4j",
    password="passw0rd",
    database="neo4j",
    embedding_dim=384,
    similarity="cosine",
    recreate_index=True,
)

# Create components and an indexing pipeline that converts txt to documents, cleans and splits them, and
# indexes them for dense retrieval.
p = Pipeline()
p.add_component("text_file_converter", TextFileToDocument())
p.add_component("cleaner", DocumentCleaner())
p.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=250, split_overlap=30))
p.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
p.add_component("writer", DocumentWriter(document_store=document_store))

p.connect("text_file_converter.documents", "cleaner.documents")
p.connect("cleaner.documents", "splitter.documents")
p.connect("splitter.documents", "embedder.documents")
p.connect("embedder.documents", "writer.documents")

# Take the docs data directory as input and run the pipeline
file_paths = [docs_dir / Path(name) for name in os.listdir(docs_dir)]
result = p.run({"text_file_converter": {"sources": file_paths}})

# Assuming you have a docker container running navigate to http://localhost:7474 to open Neo4j Browser
