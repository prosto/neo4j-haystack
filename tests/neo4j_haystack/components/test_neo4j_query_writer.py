from unittest import mock

import pytest
from haystack import Document, Pipeline, component
from haystack.dataclasses.chat_message import ChatMessage

from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
from neo4j_haystack.components.neo4j_query_writer import Neo4jQueryWriter
from neo4j_haystack.components.neo4j_retriever import Neo4jClient


@pytest.fixture
def client_config(neo4j_database: Neo4jClientConfig) -> Neo4jClientConfig:
    return neo4j_database


@pytest.fixture
def neo4j_client(client_config: Neo4jClientConfig) -> Neo4jClient:
    return Neo4jClient(config=client_config)


@pytest.mark.integration
def test_query_writer(client_config: Neo4jClientConfig, neo4j_client: Neo4jClient):
    doc_id = "t1_123"
    doc_meta = {"year": 2024, "source_url": "https://www.deepset.ai/blog"}

    writer = Neo4jQueryWriter(client_config=client_config, verify_connectivity=True, runtime_parameters=["doc_meta"])

    result = writer.run(
        query=(
            "MERGE (doc:`Document` {id: $doc_id})"
            "SET doc += {id: $doc_id, content: $content, year: $doc_meta.year, source_url: $doc_meta.source_url}"
        ),
        parameters={"doc_id": doc_id, "content": "beautiful graph"},
        doc_meta=doc_meta,
    )

    assert result["query_type"] == "w"

    records = list(
        neo4j_client.query_nodes("MATCH (doc:Document {id:$doc_id}) RETURN doc", parameters={"doc_id": doc_id})
    )
    assert len(records) == 1

    doc_in_neo4j = records[0].data().get("doc")
    assert doc_in_neo4j == {
        "id": doc_id,
        "content": "beautiful graph",
        "year": 2024,
        "source_url": "https://www.deepset.ai/blog",
    }


@pytest.mark.integration
def test_query_writer_with_document_input(client_config: Neo4jClientConfig, neo4j_client: Neo4jClient):
    doc_id = "t2_123"
    document = Document(id=doc_id, content="beautiful graph", meta={"year": 2024})

    writer = Neo4jQueryWriter(client_config=client_config, verify_connectivity=True, runtime_parameters=["document"])

    runtime_params = {"document": document}
    result = writer.run(
        query=(
            "MERGE (doc:`Document` {id: $document.id})"
            "SET doc += {id: $document.id, content: $document.content, year: $document.year}"
        ),
        **runtime_params,  # Document will be serialized with `Document.to_dict`
    )

    assert result["query_type"] == "w"

    records = list(
        neo4j_client.query_nodes("MATCH (doc:Document {id:$doc_id}) RETURN doc", parameters={"doc_id": doc_id})
    )
    assert len(records) == 1

    doc_in_neo4j = records[0].data().get("doc")
    assert Document.from_dict(doc_in_neo4j) == document


@pytest.mark.integration
def test_query_writer_with_dataclass_input(client_config: Neo4jClientConfig, neo4j_client: Neo4jClient):
    session_id = "123"
    chat_messages = [
        ChatMessage.from_system("You are are helpful chatbot"),
        ChatMessage.from_assistant("How can I help you?"),
        ChatMessage.from_user("Tell me your secret"),
    ]

    writer = Neo4jQueryWriter(
        client_config=client_config, verify_connectivity=True, runtime_parameters=["chat_messages"]
    )

    query = """
    // Create session node with a related first chat message from the list
    MERGE (session:`Session` {id: $session_id})
    WITH session
    CREATE (session)-[:FIRST_MESSAGE]->(msg_node:`ChatMessage`)
    WITH msg_node
    MATCH (msg_node) SET msg_node += $chat_messages[0]
    WITH msg_node as first_message

    // Create remaining chat message nodes
    UNWIND tail($chat_messages) as tail_message
    CREATE (msg_node:`ChatMessage`)
    WITH msg_node, tail_message, first_message
    MATCH (msg_node) SET msg_node += tail_message

    // Connect chat messages with :NEXT_MESSAGE relationship
    WITH collect(msg_node) AS message_nodes, first_message
    WITH [first_message] + message_nodes as all_nodes
    FOREACH (i IN range(0, size(all_nodes)-2)
    | FOREACH (n1 IN [all_nodes[i]]
    | FOREACH (n2 IN [all_nodes[i+1]]
    | CREATE (n1)-[:NEXT_MESSAGE]->(n2))))
    """

    runtime_params = {"chat_messages": chat_messages}
    result = writer.run(
        query=query,
        parameters={"session_id": session_id},
        **runtime_params,  # Will be serialized to list of dictionaries
    )

    assert result["query_type"] == "w"

    cypher_query_get_messages = """
    MATCH (session:Session {id:$session_id})-[:FIRST_MESSAGE|NEXT_MESSAGE*]->(message)
    RETURN session, collect(message) as messages
    """

    records = list(
        neo4j_client.query_nodes(
            cypher_query_get_messages,
            parameters={"session_id": session_id},
        )
    )
    assert len(records) == 1

    session_in_neo4j = records[0].data().get("session")
    assert session_in_neo4j == {"id": session_id}

    messages_in_neo4j = records[0].data().get("messages")
    assert messages_in_neo4j == [
        {"role": "system", "content": "You are are helpful chatbot"},
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "Tell me your secret"},
    ]


@pytest.mark.integration
def test_pipeline_execution(client_config: Neo4jClientConfig, neo4j_client: Neo4jClient):
    @component
    class YearProvider:
        @component.output_types(year_start=int, year_end=int)
        def run(self, year_start: int, year_end: int):
            return {"year_start": year_start, "year_end": year_end}

    doc_id = "001"

    query = "CREATE (doc:Document {id: $id, content: $content, year_start: $year_start, year_end: $year_end})"
    writer = Neo4jQueryWriter(
        client_config=client_config, verify_connectivity=True, runtime_parameters=["year_start", "year_end"]
    )

    pipeline = Pipeline()
    pipeline.add_component("year_provider", YearProvider())
    pipeline.add_component("writer", writer)
    pipeline.connect("year_provider.year_start", "writer.year_start")
    pipeline.connect("year_provider.year_end", "writer.year_end")

    pipeline.run(
        data={
            "year_provider": {"year_start": 2017, "year_end": 2024},
            "writer": {
                "query": query,
                "parameters": {
                    "id": doc_id,
                    "content": "Hello? Is it transformer?",
                },
            },
        }
    )

    records = list(
        neo4j_client.query_nodes("MATCH (doc:Document {id:$doc_id}) RETURN doc", parameters={"doc_id": doc_id})
    )
    assert len(records) == 1

    doc_in_neo4j = records[0].data().get("doc")
    assert doc_in_neo4j == {"id": doc_id, "content": "Hello? Is it transformer?", "year_start": 2017, "year_end": 2024}


@pytest.mark.unit
@mock.patch("neo4j_haystack.components.neo4j_query_writer.Neo4jClientConfig", spec=Neo4jClientConfig)
@mock.patch("neo4j_haystack.components.neo4j_query_writer.Neo4jClient", spec=Neo4jClient)
def test_neo4j_query_writer_to_dict(neo4j_client_mock, client_config_mock):
    neo4j_client = neo4j_client_mock.return_value  # capturing instance created in Neo4jQueryWriter
    client_config_mock.configure_mock(**{"to_dict.return_value": {"mock": "mock"}})

    retriever = Neo4jQueryWriter(
        client_config=client_config_mock,
        runtime_parameters=["year"],
        verify_connectivity=True,
    )

    data = retriever.to_dict()

    assert data == {
        "type": "neo4j_haystack.components.neo4j_query_writer.Neo4jQueryWriter",
        "init_parameters": {
            "runtime_parameters": ["year"],
            "verify_connectivity": True,
            "client_config": {"mock": "mock"},
        },
    }
    neo4j_client.verify_connectivity.assert_called_once()


@pytest.mark.unit
@mock.patch.object(Neo4jClientConfig, "from_dict")
@mock.patch("neo4j_haystack.components.neo4j_query_writer.Neo4jClient", spec=Neo4jClient)
def test_neo4j_query_writer_from_dict(neo4j_client_mock, from_dict_mock):
    neo4j_client = neo4j_client_mock.return_value  # capturing instance created in Neo4jQueryWriter
    expected_client_config = mock.Mock(spec=Neo4jClientConfig)
    from_dict_mock.return_value = expected_client_config

    data = {
        "type": "neo4j_haystack.components.neo4j_query_writer.Neo4jQueryWriter",
        "init_parameters": {
            "runtime_parameters": ["year"],
            "verify_connectivity": True,
            "client_config": {"mock": "mock"},
        },
    }

    retriever = Neo4jQueryWriter.from_dict(data)

    assert retriever._client_config == expected_client_config
    assert retriever._runtime_parameters == ["year"]
    assert retriever._verify_connectivity is True

    neo4j_client.verify_connectivity.assert_called_once()
