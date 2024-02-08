from operator import attrgetter
from typing import Callable, List

import numpy as np
import pytest
from haystack import Document

from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore


@pytest.mark.unit
def test_document_store_connection_parameters():
    skip_db_interactions_for_test = {
        "create_index_if_missing": False,
        "verify_connectivity": False,
        "recreate_index": False,
    }

    connection_config = {
        "url": "bolt://db:7687",
        "username": "username",
        "password": "password",
        "database": "database",
    }

    doc_store = Neo4jDocumentStore(**connection_config, **skip_db_interactions_for_test)

    assert doc_store.client_config.url == "bolt://db:7687"
    assert doc_store.client_config.database == "database"

    client_config = Neo4jClientConfig(**connection_config)
    doc_store = Neo4jDocumentStore(client_config=client_config, **skip_db_interactions_for_test)

    assert doc_store.client_config == client_config


@pytest.mark.integration
def test_get_all_documents_generator(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    assert len(list(doc_store.get_all_documents_generator(batch_size=2))) == 9


@pytest.mark.integration
def test_get_all_documents_without_embeddings(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    out = doc_store.get_all_documents_generator(return_embedding=False)
    for doc in out:
        assert doc.embedding is None


@pytest.mark.integration
def test_get_document_by_id(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    doc = doc_store.get_document_by_id(documents[0].id)

    assert doc
    assert doc.id == documents[0].id
    assert doc.content == documents[0].content


@pytest.mark.integration
def test_get_documents_by_id(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    ids = [doc.id for doc in documents]
    result = {doc.id for doc in doc_store.get_documents_by_id(ids, batch_size=2)}
    assert set(ids) == result


@pytest.mark.integration
def test_count_documents_with_filter(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    assert doc_store.count_documents_with_filter() == len(documents)
    assert doc_store.count_documents_with_filter({"field": "year", "operator": "in", "value": [2020]}) == 3
    assert doc_store.count_documents_with_filter({"field": "month", "operator": "==", "value": "02"}) == 3


@pytest.mark.integration
def test_get_embedding_count(doc_store: Neo4jDocumentStore, documents: List[Document]):
    """
    We expect 6 docs with embeddings because only 6 documents in the documents fixture for this class contain
    embeddings.
    """
    doc_store.write_documents(documents)
    assert doc_store.count_documents_with_filter({"field": "embedding", "operator": "exists", "value": True}) == 6


@pytest.mark.integration
def test_delete_all_documents(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    doc_store.delete_all_documents()
    assert doc_store.count_documents() == 0


@pytest.mark.integration
def test_delete_documents_with_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    doc_store.delete_all_documents(filters={"field": "year", "operator": "in", "value": [2020, 2021]})
    documents = doc_store.filter_documents()
    assert doc_store.count_documents() == 3


@pytest.mark.integration
def test_delete_documents_by_id(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    docs_to_delete = doc_store.filter_documents(filters={"field": "year", "operator": "==", "value": 2020})
    doc_store.delete_all_documents(document_ids=[doc.id for doc in docs_to_delete])
    assert doc_store.count_documents() == 6


@pytest.mark.integration
def test_delete_documents_by_id_with_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    docs_to_delete = doc_store.filter_documents(filters={"field": "year", "operator": "==", "value": 2020})
    # this should delete only 1 document out of the 3 ids passed
    doc_store.delete_all_documents(
        document_ids=[doc.id for doc in docs_to_delete], filters={"field": "name", "operator": "==", "value": "name_0"}
    )
    assert doc_store.count_documents() == 8


@pytest.mark.integration
def test_delete_index(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    assert doc_store.count_documents() == len(documents)

    doc_store.delete_index()
    assert doc_store.count_documents() == 0
    assert (
        doc_store.neo4j_client.retrieve_vector_index(doc_store.index, doc_store.node_label, doc_store.embedding_field)
        is None
    )


@pytest.mark.integration
def test_delete_index_does_not_raise_if_not_exists(doc_store_factory: Callable[..., Neo4jDocumentStore]):
    """By default Neo4j will trigger DatabaseError (server) if trying to remove index which does not exist"""
    doc_store = doc_store_factory(index="document-embeddings", create_index_if_missing=False)
    doc_store.delete_index()


@pytest.mark.integration
def test_get_all_documents_large_quantities(doc_store: Neo4jDocumentStore):
    docs_to_write = [
        Document.from_dict(
            {"meta": {"name": f"name_{i}"}, "content": f"text_{i}", "embedding": np.random.rand(768).astype(np.float32)}
        )
        for i in range(1000)
    ]
    doc_store.write_documents(docs_to_write)
    documents = doc_store.filter_documents()

    assert len(documents) == len(docs_to_write)


@pytest.mark.integration
def test_update_embeddings(
    doc_store_factory: Callable[..., Neo4jDocumentStore],
    movie_documents: List[Document],
    movie_documents_with_embeddings: List[Document],
):
    doc_store = doc_store_factory(embedding_dim=384, similarity="cosine", recreate_index=True)
    doc_store.write_documents(movie_documents)  # no embeddings at this point

    documents = doc_store.filter_documents()
    assert all(not doc.embedding for doc in documents)

    doc_store.update_embeddings(documents=movie_documents_with_embeddings)

    documents = doc_store.filter_documents()
    assert all(doc.embedding for doc in documents)


@pytest.mark.integration
def test_query_embeddings(
    doc_store_factory: Callable[..., Neo4jDocumentStore],
    movie_documents_with_embeddings: List[Document],
    text_embedder: Callable[..., List[float]],
):
    doc_store = doc_store_factory(embedding_dim=384, similarity="cosine", recreate_index=True)
    doc_store.write_documents(documents=movie_documents_with_embeddings)

    query_embedding = text_embedder(
        "A young fella pretending to be a good citizen but actually planning to commit a crime"
    )

    documents = doc_store.query_by_embedding(query_embedding, top_k=5)
    assert len(documents) == 5

    expected_content = "A film student robs a bank under the guise of shooting a short film about a bank robbery."
    retrieved_contents = list(map(attrgetter("content"), documents))
    assert expected_content in retrieved_contents

    documents = doc_store.query_by_embedding(
        query_embedding,
        top_k=5,
        filters={"field": "release_date", "operator": "==", "value": "2018-12-09"},
    )
    assert len(documents) == 1


@pytest.mark.integration
def test_update_meta(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    doc = documents[0]
    doc_store.update_document_meta(doc.id, meta={"year": 2099, "month": "12"})
    doc = doc_store.get_document_by_id(doc.id)
    assert doc.meta["year"] == 2099
    assert doc.meta["month"] == "12"
