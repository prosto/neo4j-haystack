import json
import os
import socket
import time
from typing import Any, Callable, Generator, List

import docker
import numpy as np
import pytest
from haystack import Document
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from neo4j import Driver, GraphDatabase

from neo4j_haystack.client import Neo4jClientConfig
from neo4j_haystack.document_stores import Neo4jDocumentStore

NEO4J_PORT = 7689
EMBEDDING_DIM = 768


@pytest.fixture
def documents() -> List[Document]:
    documents = []
    for i in range(3):
        documents.append(
            Document(
                content=f"A Foo Document {i}",
                meta={"name": f"name_{i}", "year": 2020, "month": "01", "numbers": [2, 4]},
                embedding=np.random.rand(EMBEDDING_DIM).astype(np.float32),
            )
        )

        documents.append(
            Document(
                content=f"A Bar Document {i}",
                meta={"name": f"name_{i}", "year": 2021, "month": "02", "numbers": [-2, -4]},
                embedding=np.random.rand(EMBEDDING_DIM).astype(np.float32),
            )
        )

        documents.append(
            Document(
                content=f"Document {i} without embeddings",
                meta={"name": f"name_{i}", "no_embedding": True, "month": "03"},
            )
        )

    return documents


@pytest.fixture
def movie_documents() -> List[Document]:
    current_test_dir = os.path.dirname(__file__)

    with open(os.path.join(current_test_dir, "./samples/movies.json")) as movies_json:
        file_contents = movies_json.read()
        docs_json = json.loads(file_contents)
        documents = [Document.from_dict(doc_json) for doc_json in docs_json]

    return documents


def _get_free_tcp_port():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    addr, port = tcp.getsockname()
    tcp.close()
    return port


def _connection_established(db_driver: Driver) -> bool:
    """
    Periodically check neo4j database connectivity and return connection status (:const:`True` if has been established)
    """
    timeout = 120
    stop_time = 3
    elapsed_time = 0
    connection_established = False
    while not connection_established and elapsed_time < timeout:
        try:
            db_driver.verify_connectivity()
            connection_established = True
        except Exception:
            time.sleep(stop_time)
            elapsed_time += stop_time
    return connection_established


@pytest.fixture(scope="module")
def neo4j_database():
    """
    Starts Neo4j docker container and waits until Neo4j database is ready.
    Returns Neo4j client configuration which represents the database in the docker container.
    Container is removed after test suite execution. The `scope` is set to ``session`` to keep only one docker
    container instance fof the whole duration of tests execution to speedup the process.
    """
    neo4j_port = _get_free_tcp_port()
    neo4j_version = os.environ.get("NEO4J_VERSION", "neo4j:5.13.0")

    config = Neo4jClientConfig(
        f"bolt://localhost:{neo4j_port}", database="neo4j", username="neo4j", password="passw0rd"
    )

    client = docker.from_env()
    container = client.containers.run(
        image=neo4j_version,
        auto_remove=True,
        environment={
            "NEO4J_AUTH": f"{config.username}/{config.password}",
        },
        name=f"test_neo4j_haystack-{neo4j_port}",
        ports={"7687/tcp": ("127.0.0.1", neo4j_port)},
        detach=True,
        remove=True,
    )

    db_driver = GraphDatabase.driver(config.url, database=config.database, auth=config.auth)

    if not _connection_established(db_driver):
        pytest.exit("Could not startup neo4j docker container and establish connection with database")

    yield config

    db_driver.close()
    container.stop()


@pytest.fixture
def doc_store_factory(neo4j_database: Neo4jClientConfig) -> Callable[..., Neo4jDocumentStore]:
    """
    A factory function to create `Neo4jDocumentStore`. It depends on the `neo4j_database` fixture (a running
    docker container with neo4j database). Can be used to construct different flavours of `Neo4jDocumentStore`
    by providing necessary initialization params through keyword arguments (see `doc_store_params`).
    """

    def _doc_store(**doc_store_params) -> Neo4jDocumentStore:
        return Neo4jDocumentStore(
            **dict(
                {
                    "url": neo4j_database.url,
                    "username": neo4j_database.username,
                    "password": neo4j_database.password,
                    "database": neo4j_database.database,
                    "embedding_dim": EMBEDDING_DIM,
                    "embedding_field": "embedding",
                    "index": "document-embeddings",
                    "node_label": "Document",
                },
                **doc_store_params,
            )
        )

    return _doc_store


@pytest.fixture
def doc_store(doc_store_factory: Callable[..., Neo4jDocumentStore]) -> Generator[Neo4jDocumentStore, Any, Any]:
    """
    A default instance of the document store to be used in tests.
    Please notice data (and index) is removed after each test execution to provide a clean state for the next test.
    """
    ds = doc_store_factory()

    yield ds

    # Remove all data from DB to start a new test with clean state
    ds.delete_index()


@pytest.fixture(scope="session")
def text_embedder() -> Callable[[str], List[float]]:
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    text_embedder.warm_up()

    def _text_embedder(text: str) -> List[float]:
        return text_embedder.run(text)["embedding"]

    return _text_embedder


@pytest.fixture(scope="session")
def doc_embedder() -> Callable[[List[Document]], List[Document]]:
    doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    doc_embedder.warm_up()

    def _doc_embedder(documents: List[Document]) -> List[Document]:
        return doc_embedder.run(documents)["documents"]

    return _doc_embedder


@pytest.fixture
def movie_documents_with_embeddings(
    movie_documents: List[Document],
    doc_embedder: Callable[[List[Document]], List[Document]],
) -> List[Document]:
    documents_copy = [Document.from_dict(doc.to_dict()) for doc in movie_documents]
    return doc_embedder(documents_copy)
