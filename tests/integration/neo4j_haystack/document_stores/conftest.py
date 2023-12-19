import json
import os
import time
from typing import Any, Callable, Generator, List

import docker
import numpy as np
import pytest
from haystack.schema import Document
from neo4j import Driver, GraphDatabase

from neo4j_haystack.document_stores.neo4j_client import Neo4jClientConfig
from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore

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
        documents = [Document.from_json(doc_json) for doc_json in docs_json]

    return documents


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


@pytest.fixture(scope="session")
def neo4j_database():
    """
    Starts neo4j docker container and waits until neo4j database is ready.
    Returns neo4j client configuration which represents the database in the docker container.
    Container is removed after test suite execution. The `scope` is set to ``session`` to keep only one docker
    container instance fof the whole duration of tests execution to speedup the process.
    """
    config = Neo4jClientConfig(
        f"bolt://localhost:{NEO4J_PORT}", database="neo4j", username="neo4j", password="passw0rd"
    )

    client = docker.from_env()
    container = client.containers.run(
        image="neo4j:5.13.0",
        auto_remove=True,
        environment={
            "NEO4J_AUTH": f"{config.username}/{config.password}",
        },
        name="test_neo4j_haystack",
        ports={"7687/tcp": ("127.0.0.1", NEO4J_PORT)},
        detach=True,
        remove=True,
    )

    db_driver = GraphDatabase.driver(config.url, database=config.database, auth=config.driver_config.get("auth"))

    if not _connection_established(db_driver):
        pytest.exit("Could not startup neo4j docker container and establish connection with database")

    yield config

    db_driver.close()
    container.stop()


@pytest.fixture
def doc_store_factory(neo4j_database: Neo4jClientConfig) -> Callable[..., Neo4jDocumentStore]:
    """
    A factory function to create :class:`Neo4jDocumentStore`. It depends on the `neo4j_database` fixture (a running
    docker container with neo4j database). Can be used to construct different flavours of :class:`Neo4jDocumentStore`
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
