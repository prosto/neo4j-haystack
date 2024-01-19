from typing import List
from unittest import mock

import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from neo4j_haystack.client import Neo4jClient, Neo4jClientConfig
from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore


@pytest.mark.integration
def test_write_documents(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    docs = doc_store.filter_documents()
    assert len(docs) == len(documents)
    assert all(isinstance(doc, Document) for doc in docs)

    expected_ids = {doc.id for doc in documents}
    ids = {doc.id for doc in docs}
    assert ids == expected_ids


@pytest.mark.integration
def test_write_with_duplicate_doc_ids(doc_store: Neo4jDocumentStore):
    duplicate_documents = [
        Document(content="Doc1", meta={"key1": "value1"}),
        Document(content="Doc1", meta={"key1": "value1"}),
    ]
    doc_store.write_documents(duplicate_documents, policy=DuplicatePolicy.SKIP)
    results = doc_store.filter_documents()

    assert len(results) == 1
    assert results[0] == duplicate_documents[0]
    with pytest.raises(DuplicateDocumentError):
        doc_store.write_documents(duplicate_documents, policy=DuplicatePolicy.FAIL)


@pytest.mark.integration
def test_count_documents(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    assert doc_store.count_documents() == len(documents)


@pytest.mark.integration
def test_filter_documents_duplicate_text_value(doc_store: Neo4jDocumentStore):
    documents = [
        Document(content="duplicated", meta={"meta_field": "0"}),
        Document(content="duplicated", meta={"meta_field": "1", "name": "file.txt"}),
        Document(content="Doc2", meta={"name": "file_2.txt"}),
    ]
    doc_store.write_documents(documents)

    documents = doc_store.filter_documents(filters={"field": "meta_field", "operator": "in", "value": ["1"]})
    assert len(documents) == 1
    assert documents[0].content == "duplicated"
    assert documents[0].meta["name"] == "file.txt"

    documents = doc_store.filter_documents(filters={"field": "meta_field", "operator": "in", "value": ["0"]})
    assert len(documents) == 1
    assert documents[0].content == "duplicated"
    assert documents[0].meta.get("name") is None

    documents = doc_store.filter_documents(filters={"field": "name", "operator": "==", "value": "file_2.txt"})
    assert len(documents) == 1
    assert documents[0].content == "Doc2"
    assert documents[0].meta.get("meta_field") is None


@pytest.mark.integration
def test_filter_documents_with_correct_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.filter_documents(filters={"field": "year", "operator": "in", "value": [2020]})
    assert len(result) == 3

    documents = doc_store.filter_documents(filters={"field": "year", "operator": "in", "value": [2020, 2021]})
    assert len(documents) == 6


@pytest.mark.integration
def test_filter_documents_with_incorrect_filter_name(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.filter_documents(
        filters={"field": "non_existing_meta_field", "operator": "in", "value": ["whatever"]}
    )
    assert len(result) == 0


@pytest.mark.integration
def test_filter_documents_with_incorrect_filter_value(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    result = doc_store.filter_documents(filters={"field": "year", "operator": "==", "value": "nope"})
    assert len(result) == 0


@pytest.mark.integration
def test_duplicate_documents_skip(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    updated_docs = []
    for doc in documents:
        updated_d = Document.from_dict(doc.to_dict())
        updated_d.meta["name"] = "Updated"
        updated_docs.append(updated_d)

    doc_store.write_documents(updated_docs, policy=DuplicatePolicy.SKIP)
    for doc in doc_store.filter_documents():
        assert doc.meta.get("name") != "Updated"


@pytest.mark.integration
def test_duplicate_documents_overwrite(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    updated_docs = []
    for doc in documents:
        updated_d = Document.from_dict(doc.to_dict())
        updated_d.meta["name"] = "Updated"
        updated_docs.append(updated_d)

    doc_store.write_documents(updated_docs, policy=DuplicatePolicy.OVERWRITE)
    for doc in doc_store.filter_documents():
        assert doc.meta["name"] == "Updated"


@pytest.mark.integration
def test_duplicate_documents_fail(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    updated_docs = []
    for doc in documents:
        updated_d = Document.from_dict(doc.to_dict())
        updated_d.meta["name"] = "Updated"
        updated_docs.append(updated_d)

    with pytest.raises(DuplicateDocumentError):
        doc_store.write_documents(updated_docs, policy=DuplicatePolicy.FAIL)


@pytest.mark.integration
def test_write_document_meta(doc_store: Neo4jDocumentStore):
    doc_store.write_documents(
        [
            Document.from_dict({"content": "dict_without_meta", "id": "1"}),
            Document.from_dict({"content": "dict_with_meta", "meta_field": "test2", "id": "2"}),
            Document(content="document_object_without_meta", id="3"),
            Document(content="document_object_with_meta", meta={"meta_field": "test4"}, id="4"),
        ]
    )

    doc1, doc2, doc3, doc4 = doc_store.filter_documents(
        filters={"field": "id", "operator": "in", "value": ["1", "2", "3", "4"]}
    )

    assert not doc1.meta
    assert doc2.meta["meta_field"] == "test2"
    assert not doc3.meta
    assert doc4.meta["meta_field"] == "test4"


@pytest.mark.integration
def test_delete_documents(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    docs_to_delete = doc_store.filter_documents(filters={"field": "year", "operator": "==", "value": 2020})
    doc_store.delete_documents(document_ids=[doc.id for doc in docs_to_delete])

    assert doc_store.count_documents() == 6


@mock.patch.object(Neo4jClientConfig, "to_dict")
@mock.patch.object(Neo4jClient, "verify_connectivity")
def test_document_store_to_dict(verify_connectivity_mock, to_dict_mock):
    to_dict_mock.return_value = {"mock": "mock"}

    doc_store = Neo4jDocumentStore(
        url="bolt://localhost:7687",
        database="database",
        username="neo4j",
        password="passw0rd",
        index="document-embeddings",
        node_label="Document",
        embedding_dim=384,
        embedding_field="embedding",
        similarity="cosine",
        progress_bar=False,
        create_index_if_missing=False,
        recreate_index=False,
        write_batch_size=111,
    )
    data = doc_store.to_dict()

    assert data == {
        "type": "neo4j_haystack.document_stores.neo4j_store.Neo4jDocumentStore",
        "init_parameters": {
            "index": "document-embeddings",
            "node_label": "Document",
            "embedding_dim": 384,
            "embedding_field": "embedding",
            "similarity": "cosine",
            "progress_bar": False,
            "create_index_if_missing": False,
            "recreate_index": False,
            "write_batch_size": 111,
            "client_config": {"mock": "mock"},
        },
    }
    verify_connectivity_mock.assert_called_once()


@pytest.mark.integration
@mock.patch.object(Neo4jClientConfig, "from_dict")
@mock.patch.object(Neo4jClient, "verify_connectivity")
def test_document_store_from_dict(verify_connectivity_mock, from_dict_mock):
    data = {
        "type": "neo4j_haystack.document_stores.neo4j_store.Neo4jDocumentStore",
        "init_parameters": {
            "index": "document-embeddings",
            "node_label": "Document",
            "embedding_dim": 384,
            "embedding_field": "embedding",
            "similarity": "cosine",
            "progress_bar": False,
            "create_index_if_missing": False,
            "recreate_index": False,
            "write_batch_size": 111,
            "client_config": {"url": "bolt://localhost:7687"},
        },
    }
    expected_client_config = Neo4jClientConfig(url="bolt://localhost:7687")
    from_dict_mock.return_value = expected_client_config

    doc_store = Neo4jDocumentStore.from_dict(data)

    assert doc_store.client_config == expected_client_config
    assert doc_store.url == "bolt://localhost:7687"
    assert doc_store.index == "document-embeddings"
    assert doc_store.node_label == "Document"
    assert doc_store.embedding_dim == 384
    assert doc_store.embedding_field == "embedding"
    assert doc_store.similarity == "cosine"
    assert doc_store.progress_bar is False
    assert doc_store.create_index_if_missing is False
    assert doc_store.recreate_index is False
    assert doc_store.write_batch_size == 111

    verify_connectivity_mock.assert_called_once()
