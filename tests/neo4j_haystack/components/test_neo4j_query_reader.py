from typing import List
from unittest import mock

import pytest
from haystack import Document

from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
from neo4j_haystack.components.neo4j_query_reader import Neo4jQueryReader
from neo4j_haystack.components.neo4j_retriever import Neo4jClient
from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore


@pytest.fixture
def client_config(
    neo4j_database: Neo4jClientConfig,
    doc_store: Neo4jDocumentStore,
    documents: List[Document],
) -> Neo4jClientConfig:
    doc_store.write_documents(documents)
    return neo4j_database


@pytest.mark.integration
def test_query_reader(client_config: Neo4jClientConfig):
    query = "MATCH (doc:`Document`) WHERE doc.year=$year RETURN doc.name as name, doc.year as year"
    reader = Neo4jQueryReader(
        client_config=client_config,
        query=query,
        verify_connectivity=True,
        runtime_parameters=["year"],
    )

    result = reader.run(year=2020)

    assert result["records"] == [
        {"name": "name_0", "year": 2020},
        {"name": "name_1", "year": 2020},
        {"name": "name_2", "year": 2020},
    ]


@pytest.mark.integration
def test_query_reader_single_record(client_config: Neo4jClientConfig):
    reader = Neo4jQueryReader(client_config=client_config, verify_connectivity=True)

    result = reader.run(
        query=("MATCH (doc:`Document` {name: $name}) RETURN doc.name as name, doc.year as year"),
        parameters={"name": "name_1"},
    )

    assert result["first_record"] == {"name": "name_1", "year": 2020}


@pytest.mark.integration
def test_query_reader_error_result(client_config: Neo4jClientConfig):
    reader = Neo4jQueryReader(client_config=client_config, raise_on_failure=False)

    result = reader.run(
        query=("MATCH (doc:`Document` {name: $name}) RETURN_ doc.name as name, doc.year as year"),
        parameters={"name": "name_1"},
    )

    assert "Invalid input 'RETURN_'" in result["error_message"]


@pytest.mark.integration
def test_query_reader_raises_error(client_config: Neo4jClientConfig):
    reader = Neo4jQueryReader(client_config=client_config, raise_on_failure=True)

    with pytest.raises(Exception):  # noqa: B017
        reader.run(
            query=("MATCH (doc:`Document` {name: $name}) RETURN_ doc.name as name, doc.year as year"),
            parameters={"name": "name_1"},
        )


@pytest.mark.unit
@mock.patch("neo4j_haystack.components.neo4j_query_reader.Neo4jClientConfig", spec=Neo4jClientConfig)
@mock.patch("neo4j_haystack.components.neo4j_query_reader.Neo4jClient", spec=Neo4jClient)
def test_neo4j_query_reader_to_dict(neo4j_client_mock, client_config_mock):
    neo4j_client = neo4j_client_mock.return_value  # capturing instance created in Neo4jQueryReader
    client_config_mock.configure_mock(**{"to_dict.return_value": {"mock": "mock"}})

    reader = Neo4jQueryReader(
        client_config=client_config_mock,
        query="cypher",
        runtime_parameters=["year"],
        verify_connectivity=True,
        raise_on_failure=False,
    )

    data = reader.to_dict()

    assert data == {
        "type": "neo4j_haystack.components.neo4j_query_reader.Neo4jQueryReader",
        "init_parameters": {
            "query": "cypher",
            "runtime_parameters": ["year"],
            "verify_connectivity": True,
            "raise_on_failure": False,
            "client_config": {"mock": "mock"},
        },
    }
    neo4j_client.verify_connectivity.assert_called_once()


@pytest.mark.unit
@mock.patch.object(Neo4jClientConfig, "from_dict")
@mock.patch("neo4j_haystack.components.neo4j_query_reader.Neo4jClient", spec=Neo4jClient)
def test_neo4j_query_reader_from_dict(neo4j_client_mock, from_dict_mock):
    neo4j_client = neo4j_client_mock.return_value  # capturing instance created in Neo4jQueryReader
    expected_client_config = mock.Mock(spec=Neo4jClientConfig)
    from_dict_mock.return_value = expected_client_config

    data = {
        "type": "neo4j_haystack.components.neo4j_query_reader.Neo4jQueryReader",
        "init_parameters": {
            "query": "cypher",
            "runtime_parameters": ["year"],
            "verify_connectivity": True,
            "raise_on_failure": False,
            "client_config": {"mock": "mock"},
        },
    }

    reader = Neo4jQueryReader.from_dict(data)

    assert reader._query == "cypher"
    assert reader._client_config == expected_client_config
    assert reader._runtime_parameters == ["year"]
    assert reader._verify_connectivity is True
    assert reader._raise_on_failure is False

    neo4j_client.verify_connectivity.assert_called_once()
