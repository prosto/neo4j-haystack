from typing import List

import pytest
from haystack import Document

from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore


@pytest.mark.integration
def test_eq_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    result = doc_store.filter_documents(filters={"field": "year", "operator": "==", "value": 2020})
    assert len(result) == 3


@pytest.mark.integration
def test_ne_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.filter_documents(filters={"field": "year", "operator": "!=", "value": 2020})

    # Neo4j teats null values (absent properties) as incomparable in logical expressions, thus documents
    # not having `year` property are ignored
    assert len(result) == 3


@pytest.mark.integration
def test_in_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    result = doc_store.filter_documents(filters={"field": "year", "operator": "in", "value": [2020, 2021, "n.a."]})
    assert len(result) == 6


@pytest.mark.integration
def test_not_in_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    """
    Neo4j does not consider properties with null values during filtering, e.g. "year NOT IN [2020, 2026]" will ignore
    documents where ``year`` is absent (or null which is equivalent in Neo4j)
    """
    doc_store.write_documents(documents)
    result = doc_store.filter_documents(filters={"field": "year", "operator": "not in", "value": [2020, 2026]})
    assert len(result) == 3

    result = doc_store.filter_documents(filters={"field": "year", "operator": "not in", "value": [2020, 2021]})
    assert len(result) == 0


@pytest.mark.integration
def test_comparison_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.filter_documents(filters={"field": "year", "operator": ">", "value": 2020})
    assert len(result) == 3

    result = doc_store.filter_documents(filters={"field": "year", "operator": ">=", "value": 2020})
    assert len(result) == 6

    result = doc_store.filter_documents(filters={"field": "year", "operator": "<", "value": 2021})
    assert len(result) == 3

    result = doc_store.filter_documents(filters={"field": "year", "operator": "<=", "value": 2021})
    assert len(result) == 6


@pytest.mark.integration
def test_compound_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.filter_documents(
        filters={
            "operator": "AND",
            "conditions": [
                {"field": "year", "operator": ">=", "value": 2020},
                {"field": "year", "operator": "<=", "value": 2021},
            ],
        }
    )
    assert len(result) == 6


@pytest.mark.integration
def test_nested_condition_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "year", "operator": ">=", "value": 2020},
            {"field": "year", "operator": "<=", "value": 2021},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "name", "operator": "in", "value": ["name_0", "name_1"]},
                    {"field": "numbers", "operator": "<", "value": 5.0},
                ],
            },
        ],
    }
    result = doc_store.filter_documents(filters=filters)
    assert len(result) == 4

    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "year", "operator": ">=", "value": 2020},
            {"field": "year", "operator": "<=", "value": 2021},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "name", "operator": "in", "value": ["name_0", "name_1"]},
                    {
                        "operator": "AND",
                        "conditions": [
                            {"field": "name", "operator": "==", "value": "name_2"},
                            {
                                "operator": "NOT",
                                "conditions": [
                                    {"field": "month", "operator": "==", "value": "01"},
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    result = doc_store.filter_documents(filters=filters)
    assert len(result) == 5
