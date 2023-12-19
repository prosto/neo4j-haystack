from operator import attrgetter
from typing import Callable, List

import numpy as np
import pytest
from haystack.errors import DuplicateDocumentError
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document

from neo4j_haystack.document_stores.neo4j_store import Neo4jDocumentStore


@pytest.mark.integration
def test_write_documents(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    docs = doc_store.get_all_documents()
    assert len(docs) == len(documents)

    expected_ids = {doc.id for doc in documents}
    ids = {doc.id for doc in docs}
    assert ids == expected_ids


@pytest.mark.integration
def test_write_with_duplicate_doc_ids(doc_store: Neo4jDocumentStore):
    duplicate_documents = [
        Document(content="Doc1", id_hash_keys=["content"], meta={"key1": "value1"}),
        Document(content="Doc1", id_hash_keys=["content"], meta={"key1": "value1"}),
    ]
    doc_store.write_documents(duplicate_documents, duplicate_documents="skip")
    results = doc_store.get_all_documents()

    assert len(results) == 1
    assert results[0] == duplicate_documents[0]
    with pytest.raises(DuplicateDocumentError):
        doc_store.write_documents(duplicate_documents, duplicate_documents="fail")


@pytest.mark.integration
def test_get_embedding_count(doc_store: Neo4jDocumentStore, documents: List[Document]):
    """
    We expect 6 docs with embeddings because only 6 documents in the documents fixture for this class contain
    embeddings.
    """
    doc_store.write_documents(documents)
    assert doc_store.get_embedding_count() == 6


@pytest.mark.integration
def test_get_all_documents_without_embeddings(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    out = doc_store.get_all_documents(return_embedding=False)
    for doc in out:
        assert doc.embedding is None


@pytest.mark.integration
def test_get_all_document_filter_duplicate_text_value(doc_store: Neo4jDocumentStore):
    documents = [
        Document(content="duplicated", meta={"meta_field": "0"}, id_hash_keys=["meta"]),
        Document(content="duplicated", meta={"meta_field": "1", "name": "file.txt"}, id_hash_keys=["meta"]),
        Document(content="Doc2", meta={"name": "file_2.txt"}, id_hash_keys=["meta"]),
    ]
    doc_store.write_documents(documents)
    documents = doc_store.get_all_documents(filters={"meta_field": ["1"]})
    assert len(documents) == 1
    assert documents[0].content == "duplicated"
    assert documents[0].meta["name"] == "file.txt"

    documents = doc_store.get_all_documents(filters={"meta_field": ["0"]})
    assert len(documents) == 1
    assert documents[0].content == "duplicated"
    assert documents[0].meta.get("name") is None

    documents = doc_store.get_all_documents(filters={"name": ["file_2.txt"]})
    assert len(documents) == 1
    assert documents[0].content == "Doc2"
    assert documents[0].meta.get("meta_field") is None


@pytest.mark.integration
def test_get_all_documents_with_correct_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    result = doc_store.get_all_documents(filters={"year": [2020]})
    assert len(result) == 3

    documents = doc_store.get_all_documents(filters={"year": [2020, 2021]})
    assert len(documents) == 6


@pytest.mark.integration
def test_get_all_documents_with_incorrect_filter_name(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    result = doc_store.get_all_documents(filters={"non_existing_meta_field": ["whatever"]})
    assert len(result) == 0


@pytest.mark.integration
def test_get_all_documents_with_incorrect_filter_value(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    result = doc_store.get_all_documents(filters={"year": ["nope"]})
    assert len(result) == 0


@pytest.mark.integration
def test_eq_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.get_all_documents(filters={"year": {"$eq": 2020}})
    assert len(result) == 3
    result = doc_store.get_all_documents(filters={"year": 2020})
    assert len(result) == 3


@pytest.mark.integration
def test_in_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.get_all_documents(filters={"year": {"$in": [2020, 2021, "n.a."]}})
    assert len(result) == 6
    result = doc_store.get_all_documents(filters={"year": [2020, 2021, "n.a."]})
    assert len(result) == 6


@pytest.mark.integration
def test_ne_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.get_all_documents(filters={"year": {"$ne": 2020}})

    # neo4j teats null values (absent properties) as incomparable in logical expressions, thus documents
    # not having `year` property are ignored
    assert len(result) == 3


@pytest.mark.integration
def test_nin_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    """
    neo4j does not consider properties with null values during filtering, e.g. "year NOT IN [2020, 2026]" will ignore
    documents where ``year`` is absent (or null which is equivalent in neo4j)
    """
    doc_store.write_documents(documents)

    result = doc_store.get_all_documents(filters={"year": {"$nin": [2020, 2026]}})
    assert len(result) == 3


@pytest.mark.integration
def test_nin_filters_list_values(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.get_all_documents(filters={"year": {"$nin": [2020, 2026]}})
    assert len(result) == 3

    result = doc_store.get_all_documents(filters={"year": {"$nin": [2020, 2021]}})
    assert len(result) == 0


@pytest.mark.integration
def test_comparison_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.get_all_documents(filters={"year": {"$gt": 2020}})
    assert len(result) == 3

    result = doc_store.get_all_documents(filters={"year": {"$gte": 2020}})
    assert len(result) == 6

    result = doc_store.get_all_documents(filters={"year": {"$lt": 2021}})
    assert len(result) == 3

    result = doc_store.get_all_documents(filters={"year": {"$lte": 2021}})
    assert len(result) == 6


@pytest.mark.integration
def test_compound_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    result = doc_store.get_all_documents(filters={"year": {"$lte": 2021, "$gte": 2020}})
    assert len(result) == 6


@pytest.mark.integration
def test_simplified_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    filters = {"$and": {"year": {"$lte": 2021, "$gte": 2020}, "name": {"$in": ["name_0", "name_1"]}}}
    result = doc_store.get_all_documents(filters=filters)
    assert len(result) == 4

    filters_simplified = {"year": {"$lte": 2021, "$gte": 2020}, "name": ["name_0", "name_1"]}
    result = doc_store.get_all_documents(filters=filters_simplified)
    assert len(result) == 4


@pytest.mark.integration
def test_nested_condition_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    filters = {
        "$and": {
            "year": {"$lte": 2021, "$gte": 2020},
            "$or": {"name": {"$in": ["name_0", "name_1"]}, "numbers": {"$lt": 5.0}},
        }
    }
    result = doc_store.get_all_documents(filters=filters)
    assert len(result) == 4

    filters_simplified = {
        "year": {"$lte": 2021, "$gte": 2020},
        "$or": {"name": {"$in": ["name_0", "name_2"]}, "numbers": {"$lt": 5.0}},
    }
    result = doc_store.get_all_documents(filters=filters_simplified)
    assert len(result) == 4

    filters = {
        "$and": {
            "year": {"$lte": 2021, "$gte": 2020},
            "$or": {
                "name": {"$in": ["name_0", "name_1"]},
                "$and": {"name": {"$eq": "name_2"}, "$not": {"month": {"$eq": "01"}}},
            },
        }
    }
    result = doc_store.get_all_documents(filters=filters)
    assert len(result) == 5

    filters_simplified = {
        "year": {"$lte": 2021, "$gte": 2020},
        "$or": {"name": ["name_0", "name_1"], "$and": {"name": "name_2", "$not": {"month": "01"}}},
    }
    result = doc_store.get_all_documents(filters=filters_simplified)
    assert len(result) == 5


@pytest.mark.integration
def test_nested_condition_not_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    filters = {
        "$not": {
            "$or": {
                "$and": {"numbers": {"$lt": 5.0}, "month": {"$ne": "01"}},
                "$not": {"year": {"$lte": 2021, "$gte": 2020}},
            }
        }
    }
    result = doc_store.get_all_documents(filters=filters)
    assert len(result) == 3

    docs_meta = result[0].meta["numbers"]
    assert [2, 4] == docs_meta


@pytest.mark.integration
def test_same_logical_operator_twice_on_same_level(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    filters = {
        "$or": [
            {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$gte": 2020}}},
            {"$and": {"name": {"$in": ["name_0", "name_1"]}, "year": {"$lt": 2021}}},
        ]
    }
    result = doc_store.get_all_documents(filters=filters)
    docs_meta = [doc.meta["name"] for doc in result]
    assert len(result) == 4
    assert "name_0" in docs_meta
    assert "name_1" in docs_meta
    assert "name_2" not in docs_meta


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
def test_get_document_count(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    assert doc_store.get_document_count() == len(documents)
    assert doc_store.get_document_count(filters={"year": [2020]}) == 3
    assert doc_store.get_document_count(filters={"month": ["02"]}) == 3


@pytest.mark.integration
def test_get_all_documents_generator(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    assert len(list(doc_store.get_all_documents_generator(batch_size=2))) == 9


@pytest.mark.integration
def test_duplicate_documents_skip(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    updated_docs = []
    for doc in documents:
        updated_d = Document.from_dict(doc.to_dict())
        updated_d.meta["name"] = "Updated"
        updated_docs.append(updated_d)

    doc_store.write_documents(updated_docs, duplicate_documents="skip")
    for doc in doc_store.get_all_documents():
        assert doc.meta.get("name") != "Updated"


@pytest.mark.integration
def test_duplicate_documents_overwrite(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)

    updated_docs = []
    for doc in documents:
        updated_d = Document.from_dict(doc.to_dict())
        updated_d.meta["name"] = "Updated"
        updated_docs.append(updated_d)

    doc_store.write_documents(updated_docs, duplicate_documents="overwrite")
    for doc in doc_store.get_all_documents():
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
        doc_store.write_documents(updated_docs, duplicate_documents="fail")


@pytest.mark.integration
def test_write_document_meta(doc_store: Neo4jDocumentStore):
    doc_store.write_documents(
        [
            {"content": "dict_without_meta", "id": "1"},
            {"content": "dict_with_meta", "meta_field": "test2", "id": "2"},
            Document(content="document_object_without_meta", id="3"),
            Document(content="document_object_with_meta", meta={"meta_field": "test4"}, id="4"),
        ]
    )

    doc1, doc2, doc3, doc4 = doc_store.get_documents_by_id(["1", "2", "3", "4"])

    assert not doc1.meta
    assert doc2.meta["meta_field"] == "test2"
    assert not doc3.meta
    assert doc4.meta["meta_field"] == "test4"


@pytest.mark.integration
def test_delete_documents(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    doc_store.delete_documents()
    assert doc_store.get_document_count() == 0


@pytest.mark.integration
def test_delete_documents_with_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    doc_store.delete_documents(filters={"year": [2020, 2021]})
    documents = doc_store.get_all_documents()
    assert doc_store.get_document_count() == 3


@pytest.mark.integration
def test_delete_documents_by_id(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    docs_to_delete = doc_store.get_all_documents(filters={"year": [2020]})
    doc_store.delete_documents(ids=[doc.id for doc in docs_to_delete])
    assert doc_store.get_document_count() == 6


@pytest.mark.integration
def test_delete_documents_by_id_with_filters(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    docs_to_delete = doc_store.get_all_documents(filters={"year": [2020]})
    # this should delete only 1 document out of the 3 ids passed
    doc_store.delete_documents(ids=[doc.id for doc in docs_to_delete], filters={"name": ["name_0"]})
    assert doc_store.get_document_count() == 8


@pytest.mark.integration
def test_delete_index(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    assert doc_store.get_document_count() == len(documents)

    doc_store.delete_index()
    assert doc_store.get_document_count() == 0
    assert (
        doc_store.neo4j_client.retrieve_vector_index(doc_store.index, doc_store.node_label, doc_store.embedding_field)
        is None
    )


@pytest.mark.integration
def test_delete_index_does_not_raise_if_not_exists(doc_store_factory: Callable[..., Neo4jDocumentStore]):
    """By default neo4j will trigger DatabaseError (server) if trying to remove index which does not exist"""
    doc_store = doc_store_factory(create_index_if_missing=False)
    doc_store.delete_index(index=doc_store.index)


@pytest.mark.integration
def test_update_meta(doc_store: Neo4jDocumentStore, documents: List[Document]):
    doc_store.write_documents(documents)
    doc = documents[0]
    doc_store.update_document_meta(doc.id, meta={"year": 2099, "month": "12"})
    doc = doc_store.get_document_by_id(doc.id)
    assert doc.meta["year"] == 2099
    assert doc.meta["month"] == "12"


@pytest.mark.integration
def test_get_all_documents_large_quantities(doc_store: Neo4jDocumentStore):
    docs_to_write = [
        {"meta": {"name": f"name_{i}"}, "content": f"text_{i}", "embedding": np.random.rand(768).astype(np.float32)}
        for i in range(1000)
    ]
    doc_store.write_documents(docs_to_write)
    documents = doc_store.get_all_documents()
    assert all(isinstance(d, Document) for d in documents)
    assert len(documents) == len(docs_to_write)


@pytest.mark.integration
def test_custom_embedding_field(doc_store_factory: Callable[..., Neo4jDocumentStore]):
    doc_store = doc_store_factory(embedding_field="custom_embedding_field")
    custom_embedding = np.random.rand(768).astype(np.float32)
    doc_to_write = {"content": "test", "custom_embedding_field": custom_embedding}

    doc_store.write_documents([doc_to_write])
    documents = doc_store.get_all_documents(return_embedding=True)

    assert len(documents) == 1
    assert documents[0].content == "test"

    assert custom_embedding.shape == documents[0].embedding.shape


@pytest.mark.integration
def test_query_embeddings(doc_store_factory: Callable[..., Neo4jDocumentStore], movie_documents: List[Document]):
    doc_store = doc_store_factory(embedding_dim=384, similarity="cosine", recreate_index=True)
    doc_store.write_documents(movie_documents)

    retriever = EmbeddingRetriever(
        document_store=doc_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        model_format="sentence_transformers",
        use_gpu=False,
    )

    doc_store.update_embeddings(retriever)
    documents = retriever.retrieve(
        "A young fella pretending to be a good citizen but actually planning to commit a crime", top_k=5
    )

    assert len(documents) == 5

    expected_content = "A film student robs a bank under the guise of shooting a short film about a bank robbery."
    retrieved_contents = list(map(attrgetter("content"), documents))
    assert expected_content in retrieved_contents

    documents = retriever.retrieve(
        "A young fella pretending to be a good citizen but actually planning to commit a crime",
        top_k=5,
        filters={"release_date": "2018-12-09"},
    )
    assert len(documents) == 1
