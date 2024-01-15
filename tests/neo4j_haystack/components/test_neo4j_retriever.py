from operator import attrgetter
from typing import Callable, List
from unittest import mock

import pytest
from haystack import Document

from neo4j_haystack.components.neo4j_retriever import Neo4jDocumentRetriever
from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore


@pytest.fixture
def movie_document_store(
    doc_store_factory: Callable[..., Neo4jDocumentStore],
    movie_documents_with_embeddings: List[Document],
) -> Neo4jDocumentStore:
    doc_store = doc_store_factory(embedding_dim=384, similarity="cosine", recreate_index=True)
    doc_store.write_documents(movie_documents_with_embeddings)

    return doc_store


def test_retrieve_documents(movie_document_store: Neo4jDocumentStore, text_embedder: Callable[[str], List[float]]):
    retriever = Neo4jDocumentRetriever(document_store=movie_document_store)

    query_embedding = text_embedder(
        "A young fella pretending to be a good citizen but actually planning to commit a crime"
    )

    documents = retriever.run(
        query_embedding,
        top_k=5,
    )["documents"]
    assert len(documents) == 5

    expected_content = "A film student robs a bank under the guise of shooting a short film about a bank robbery."
    retrieved_contents = list(map(attrgetter("content"), documents))
    assert expected_content in retrieved_contents


@pytest.mark.integration
def test_retrieve_documents_with_filters(
    movie_document_store: Neo4jDocumentStore, text_embedder: Callable[[str], List[float]]
):
    retriever = Neo4jDocumentRetriever(document_store=movie_document_store)

    query_embedding = text_embedder(
        "A young fella pretending to be a good citizen but actually planning to commit a crime"
    )

    documents = retriever.run(
        query_embedding,
        top_k=5,
        filters={"field": "release_date", "operator": "==", "value": "2018-12-09"},
    )["documents"]
    assert len(documents) == 1


@pytest.mark.integration
def test_retriever_to_dict():
    doc_store = mock.create_autospec(Neo4jDocumentStore)
    doc_store.to_dict.return_value = {"ds": "yes"}

    retriever = Neo4jDocumentRetriever(
        document_store=doc_store,
        filters={"field": "num", "operator": ">", "value": 10},
        top_k=11,
        scale_score=False,
        return_embedding=True,
    )
    data = retriever.to_dict()

    assert data == {
        "type": "neo4j_haystack.components.neo4j_retriever.Neo4jDocumentRetriever",
        "init_parameters": {
            "document_store": {"ds": "yes"},
            "filters": {"field": "num", "operator": ">", "value": 10},
            "top_k": 11,
            "scale_score": False,
            "return_embedding": True,
        },
    }


@pytest.mark.integration
@mock.patch.object(Neo4jDocumentStore, "from_dict")
def test_retriever_from_dict(from_dict_mock):
    data = {
        "type": "neo4j_haystack.components.neo4j_retriever.Neo4jDocumentRetriever",
        "init_parameters": {
            "document_store": {"ds": "yes"},
            "filters": {"field": "num", "operator": ">", "value": 10},
            "top_k": 11,
            "scale_score": False,
            "return_embedding": True,
        },
    }
    doc_store = mock.create_autospec(Neo4jDocumentStore)
    from_dict_mock.return_value = doc_store

    retriever = Neo4jDocumentRetriever.from_dict(data)

    assert retriever._document_store == doc_store
    assert retriever._filters == {"field": "num", "operator": ">", "value": 10}
    assert retriever._top_k == 11
    assert retriever._scale_score is False
    assert retriever._return_embedding is True
