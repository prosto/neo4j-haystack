from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
from neo4j_haystack.components.neo4j_query_writer import Neo4jQueryWriter

# Make sure you have a running Neo4j database, e.g. with Docker:
# docker run \
#     --restart always \
#     --publish=7474:7474 --publish=7687:7687 \
#     --env NEO4J_AUTH=neo4j/passw0rd \
#     neo4j:5.16.0

client_config = Neo4jClientConfig(
    url="bolt://localhost:7687",
    username="neo4j",
    password="passw0rd",
    database="neo4j",
)

pipe = Pipeline()
pipe.add_component("text_file_converter", TextFileToDocument())
pipe.add_component("cleaner", DocumentCleaner())
pipe.add_component("parent_splitter", DocumentSplitter(split_by="word", split_length=512, split_overlap=30))
pipe.add_component("child_splitter", DocumentSplitter(split_by="word", split_length=100, split_overlap=24))
pipe.add_component("embedder", SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"))
pipe.add_component(
    "neo4j_writer",
    Neo4jQueryWriter(client_config=client_config, runtime_parameters=["child_documents", "parent_documents"]),
)
pipe.add_component(
    "neo4j_vector_index", Neo4jQueryWriter(client_config=client_config, runtime_parameters=["query_status"])
)

pipe.connect("text_file_converter.documents", "cleaner.documents")
pipe.connect("cleaner.documents", "parent_splitter.documents")
pipe.connect("parent_splitter.documents", "child_splitter.documents")
pipe.connect("child_splitter.documents", "embedder.documents")
pipe.connect("embedder.documents", "neo4j_writer.child_documents")
pipe.connect("parent_splitter.documents", "neo4j_writer.parent_documents")
pipe.connect("neo4j_writer.query_status", "neo4j_vector_index.query_status")

# Take the docs data directory as input and run the pipeline
file_paths = [Path(__file__).resolve().parent / "dune.txt"]

cypher_query_create_documents = """
    // Creating Parent documents
    UNWIND $parent_documents AS parent_doc
    MERGE (parent:`Document` {id: parent_doc.id})
    SET parent += parent_doc{.*, embedding: null}

    // Creating Child documents for a given 'parent' document
    WITH parent
    UNWIND $child_documents AS child_doc
    WITH parent, child_doc WHERE child_doc.source_id = parent.id
    MERGE (child:`Chunk` {id: child_doc.id})-[:HAS_PARENT]->(parent)
    SET child += child_doc{.*, embedding: null}
    WITH child, child_doc
    CALL { WITH child, child_doc
        MATCH(child:`Chunk` {id: child_doc.id}) WHERE child_doc.embedding IS NOT NULL
        CALL db.create.setNodeVectorProperty(child, 'embedding', child_doc.embedding)
    }
"""

cypher_query_create_index = """
    CREATE VECTOR INDEX `chunk-embeddings` IF NOT EXISTS
    FOR (child_doc:Chunk)
    ON (child_doc.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: $vector_dimensions,
        `vector.similarity_function`: $vector_similarity_function
    }}
"""

result = pipe.run(
    {
        "text_file_converter": {"sources": file_paths},
        "neo4j_writer": {
            "query": cypher_query_create_documents,
            "parameters": {"parent_label": "Parent", "child_label": "Chunk"},
        },
        "neo4j_vector_index": {
            "query": cypher_query_create_index,
            "parameters": {
                "vector_dimensions": 384,
                "vector_similarity_function": "cosine",
            },
        },
    }
)

# Assuming you have a docker container running navigate to http://localhost:7474 to open Neo4j Browser
