import logging
from typing import Any, ClassVar, Dict, Generator, List, Literal, Optional, Set

import numpy as np
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from neo4j.exceptions import DatabaseError
from tqdm import tqdm

from neo4j_haystack.client import Neo4jClient, Neo4jClientConfig, Neo4jRecord
from neo4j_haystack.document_stores.utils import get_batches_from_generator
from neo4j_haystack.metadata_filter import COMPARISON_OPS, FilterParser, OperatorAST

logger = logging.getLogger(__name__)


SimilarityFunction = Literal["cosine", "euclidean"]

FilterType = Dict[str, Any]


class Neo4jDocumentStore:
    """
    Document store for [Neo4j Database](https://neo4j.com/) with support for dense retrievals using \
    [Vector Search Index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)

    The implementation is based on [Python Driver](https://neo4j.com/docs/python-manual/current/) for database access.
    Document properties are stored as graph nodes. Embeddings are stored as part of node properties along with the rest
    of attributes (including meta):

    ```py title="Document json representation (e.g. `Document.to_json`)"
    {
      "id": "793764",
      "content": "Aliens and UFOs are more real than ever before...",
      "embedding": [...],
      "score": null,
      "meta": {
        "title": "Alien Chronicles Top Ufo Encounters",
        "runtime": 70.0
      }
    }
    ```

    The following should be expected after writing documents to Neo4j (see `Neo4jDocumentStore.write_documents`).

    ```json title="Neo4j node json representation for a document (with comments)"
    {
      "identity": 18900, // Neo4j native id
      "labels": ["Document"], // by default using "Document" label for the node
      "properties": {
        "id": "793764",
        "content": "Aliens and UFOs are more real than ever before...",
        "embedding": [...],

        // Document.meta fields (same level as rest of attributes)
        "title": "Alien Chronicles Top Ufo Encounters",
        "runtime": 70.0
      },
      "elementId": "18900"
    }
    ```

    Please notice the `embedding` property which is stored as part of Neo4j node properties. It has type ``LIST<FLOAT>``
    and is assigned to the node using ``db.create.setNodeVectorProperty`` procedure. The node acts as a storage for the
    `embedding` but the actual dense retrieval is performed against a dedicated search index created automatically by
    `Neo4jDocumentStore`. The index is created using `db.index.vector.createNodeIndex()` Neo4j procedure and is based
    on the `embedding` property.

    Embedding dimension as well as similarity function (e.g. `cosine`) are configurable.
    At the moment Neo4j supports only cosine and euclidean(l2) similarity functions.

    Metadata filtering by `Neo4jDocumentStore` is performed using the standard `WHERE` Cypher query clause.
    Vector search is implemented by calling `db.index.vector.queryNodes()` procedure. **Neo4j currently does not support
    metadata "pre-filtering" which runs in combination with vector search. First, vector search takes place and metadata
    is filtered based on its results.**

    The metadata filtering can be further improved by creating/tweaking \
    [Indexes for search performance](https://neo4j.com/docs/cypher-manual/current/indexes-for-search-performance/).
    It can be managed directly in Neo4j as an administrative task.

    You have several options available for deploying/installing Neo4j. See more details in \
    [Installation Operations Manual](https://neo4j.com/docs/operations-manual/current/installation/).
    As of Neo4j 5.13, the vector search index is no longer a beta feature.

    Bellow is an example how document store can be created:

    ```python
    # Obtain list of documents - there are many options available in Haystack
    documents: List[Document] = ...

    # Create `Neo4jDocumentStore` with required credentials and Vector index configuration
    document_store = Neo4jDocumentStore(
        url="bolt://localhost:7687",
        username="neo4j",
        password="passw0rd",
        database="neo4j",
        embedding_dim=384,
        index="document-embeddings", # The name of the Vector search index in Neo4j
        node_label="Document", # Providing a label to Neo4j nodes which store Documents
    )

    # Write documents to Neo4j. Respective nodes will be created.
    document_store.write_documents(documents)
    ```
    """

    SIMILARITY_MAP: ClassVar[Dict[str, SimilarityFunction]] = {
        "cosine": "cosine",
        "l2": "euclidean",
    }

    def __init__(
        self,
        url: str,
        database: Optional[str] = "neo4j",
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_config: Optional[Neo4jClientConfig] = None,
        index: str = "document-embeddings",
        node_label: str = "Document",
        embedding_dim: int = 768,
        embedding_field: str = "embedding",
        similarity: str = "cosine",
        progress_bar: bool = False,
        create_index_if_missing: Optional[bool] = True,
        recreate_index: Optional[bool] = False,
        write_batch_size: int = 100,
    ):
        """
        Constructor method

        Args:
            url: URI pointing to Neo4j instance see (https://neo4j.com/docs/api/python-driver/current/api.html#uri)
            database: Neo4j database to interact with.
            username: Username to authenticate with the database.
            password: Password credential for the given username.
            client_config: Advanced client configuration to control various settings of underlying neo4j python
                driver. See `Neo4jClientConfig` for more details. The mandatory `url` attribute will be set on
                the `client_config` in case it was provided in the config itself.
            index: The name of Neo4j Vector Search Index used for storing and querying embeddings.
            node_label: The name of the label used in Neo4j to represent `haystack.Document`.
                Neo4j nodes are used primarily as storage for Document attributes and metadata filtering.
                The filtering process includes `node_label` in database queries (e.g.
                `:::cypher MATCH (doc:<node_label>) RETURN doc`). Together with the `self.index` it identifies where
                documents are located in the database.
            embedding_dim: embedding dimension specified for the Vector search index.
            embedding_field: the name of embedding field which is created as a Neo4j node property containing an
                embedding vector. By default it is the same as in `haystack.schema.Document`. It is used during
                index creation and querying embeddings.
            similarity: similarity function specified during Vector search index creation. Supported values are
                "cosine" and "l2".
            progress_bar: Shows a tqdm progress bar.
            create_index_if_missing: Will create vector index during class initialization if it is not yet available
                in the `database`. Will only take effect if `recreate_index` is not `True`.
            recreate_index: If `True` will delete existing index and its data (documents) and create a new
                index. Useful for testing purposes when a new DocumentStore initializes with a clean database state.
            write_batch_size: Number of documents to write at once. When working with large number of documents
                batching can help reduce memory footprint.

        Raises:
            ValueError: In case similarity function specified is not supported
        """

        super().__init__()

        self.url = url
        self.index = index
        self.node_label = node_label
        self.embedding_dim = embedding_dim
        self.embedding_field = embedding_field

        self.similarity = similarity
        self.similarity_function = self._get_distance(similarity)

        self.progress_bar = progress_bar
        self.create_index_if_missing = create_index_if_missing
        self.recreate_index = recreate_index
        self.write_batch_size = write_batch_size

        self.filter_parser = FilterParser()

        if client_config and not client_config.url:
            client_config.url = url
        self.client_config = client_config or Neo4jClientConfig(url, database, username, password)
        self.neo4j_client = Neo4jClient(self.client_config)
        self.neo4j_client.verify_connectivity()

        if recreate_index:
            self.delete_index()
            self.create_index()
        elif create_index_if_missing:
            self.neo4j_client.create_index_if_missing(
                self.index, self.node_label, self.embedding_field, self.embedding_dim, self.similarity_function
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        data = default_to_dict(
            self,
            index=self.index,
            node_label=self.node_label,
            embedding_dim=self.embedding_dim,
            embedding_field=self.embedding_field,
            similarity=self.similarity,
            progress_bar=self.progress_bar,
            create_index_if_missing=self.create_index_if_missing,
            recreate_index=self.recreate_index,
            write_batch_size=self.write_batch_size,
        )

        data["init_parameters"]["client_config"] = self.client_config.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neo4jDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        client_config = Neo4jClientConfig.from_dict(data["init_parameters"]["client_config"])
        data["init_parameters"]["client_config"] = client_config
        data["init_parameters"]["url"] = client_config.url

        return default_from_dict(cls, data)

    def count_documents(self) -> int:
        """
        Returns the number of documents stored.
        """
        return self.count_documents_with_filter()

    def filter_documents(self, filters: Optional[FilterType] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Args:
            filters: Optional filters to narrow down the documents which should be returned.
                Learn more about filtering syntax in [Metadata Filtering](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        Returns:
            A list of found documents.
        """
        return list(self.get_all_documents_generator(filters, return_embedding=True))

    def write_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = DuplicatePolicy.NONE,
    ):
        """
        Writes documents to the DocumentStore.

        Args:
            documents: List of `haystack.Document`. If they already contain the embeddings, we'll index
                them right away in Neo4j. If not, you can later call `update_embeddings` to create and index them.
            policy: Handle duplicates document based on parameter options. Parameter options:
                - skip: Ignore the duplicates documents.
                - overwrite: Update any existing documents with the same ID when adding documents.
                - fail: An error is raised if the document ID of the document being added already exists

        Raises:
            DuplicateDocumentError: Exception triggers on duplicate document.
            ValueError: If `documents` parameter is not a list of of type `haystack.Document`.
        """

        for doc in documents:
            if not isinstance(doc, Document):
                msg = f"DocumentStore.write_documents() expects a list of Documents but got an element of {type(doc)}."
                raise ValueError(msg)

        if len(documents) == 0:
            logger.warning("Calling Neo4jDocumentStore.write_documents() with an empty list")
            return

        batch_size = self.write_batch_size
        document_objects = self._handle_duplicate_documents(documents, policy)

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(
            total=len(document_objects),
            desc=f"Write Documents<index: {self.index},node_label: {self.node_label}>",
            unit=" docs",
            disable=not self.progress_bar,
        ) as progress_bar:
            for document_batch in batched_documents:
                records = [self._document_to_neo4j_record(doc) for doc in document_batch]
                embedding_field = self.embedding_field
                self.neo4j_client.merge_nodes(self.node_label, embedding_field, records)
                progress_bar.update(batch_size)

    def delete_documents(self, document_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the DocumentStore.
        Fails with `MissingDocumentError` if no document with this id is present in the DocumentStore.

        Args:
            document_ids: Document ids of documents to be removed.
        """
        self.delete_all_documents(document_ids)

    def update_embeddings(
        self,
        documents: List[Document],
        batch_size: int = 100,
    ):
        """
        Updates the embeddings in the document store for given `documents`.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the embedder
        configuration).

        Args:
            documents: Documents with non-null embeddings to be updated.
            batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """

        document_count = len(documents)
        logger.debug("Updating embeddings for %s docs...", document_count)

        with tqdm(
            total=document_count, disable=not self.progress_bar, unit=" docs", desc="Updating embeddings"
        ) as progress_bar:
            for document_batch in get_batches_from_generator(documents, batch_size):
                only_embeddings = [{"id": doc.id, self.embedding_field: doc.embedding} for doc in document_batch]

                self.neo4j_client.update_embedding(
                    self.node_label,
                    self.embedding_field,
                    only_embeddings,
                )

                progress_bar.update(batch_size)

    def get_all_documents_generator(
        self,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 1000,
    ) -> Generator[Document, None, None]:
        """
        Get all documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory. Such mechanism is natively
        supported by underlying Neo4j Python Driver (an internal buffer which is depleted while being read and filled
        up while data is coming from the database)

        Args:
            filters: Optional filters to narrow down the documents which should be returned.
                Learn more about filtering syntax in [Metadata Filtering](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).
            return_embedding: To return document embedding. By default is `None` which should reduce amount of data
                returned (considering embeddings are usually large in size)
            batch_size: When working with large number of documents, batching can help reduce memory footprint.
                This parameter controls how many documents are retrieved at once from Neo4j.

        Returns:
            A Generator of found documents.
        """

        filter_ast = self._parse_filters(filters=filters)
        skip_properties = [] if return_embedding else [self.embedding_field]

        records = self.neo4j_client.find_nodes(self.node_label, filter_ast, skip_properties, fetch_size=batch_size)

        return (self._neo4j_record_to_document(rec) for rec in records)

    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieves a document by its `id`.

        Args:
            document_id: id of the Document to retrieve.

        Returns:
            A found document with matching `document_id` if exactly one is found, otherwise `None` is returned
        """

        records = self.get_documents_by_id([document_id])
        number_found = len(records)

        if number_found > 1:
            logger.warn(
                f"get_document_by_id: Found more than one document for a given id(`{id}`). "
                "Expected: 1, Found: {number_found}. Please make sure your data has unique ids"
            )

        return records[0] if number_found > 0 else None

    def get_documents_by_id(
        self,
        document_ids: List[str],
        batch_size: int = 1_000,
    ) -> List[Document]:
        """
        Retrieves all documents using their ids.

        Args:
            document_ids: List of ids to retrieve.
            batch_size: Number of documents to retrieve at a time. When working with large number of documents,
                batching can help reduce memory footprint.

        Returns:
            List of found Documents.
        """

        documents: List[Document] = []
        for batch_ids in get_batches_from_generator(document_ids, batch_size):
            filter_ast = self.filter_parser.comparison_op("id", COMPARISON_OPS.OP_IN, list(batch_ids))
            records = self.neo4j_client.find_nodes(self.node_label, filter_ast)
            documents.extend([self._neo4j_record_to_document(rec) for rec in records])

        return documents

    def count_documents_with_filter(
        self,
        filters: Optional[FilterType] = None,
    ) -> int:
        """
        Return the count of filtered documents in the document store.

        Args:
            filters: Narrow down the documents which should be counted.
                Learn more about filtering syntax in [Metadata Filtering](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

        Returns:
            Found documents count with respective filters applied.
        """

        filter_ast = self._parse_filters(filters=filters)

        return self.neo4j_client.count_nodes(self.node_label, filter_ast)

    def query_by_embedding(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        return_embedding: Optional[bool] = None,
        scale_score: bool = True,
        expand_top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        Args:
            query_embedding: Embedding of the query (e.g. gathered from Dense Retrievers)
            filters: Optional filters to narrow down the documents which should be returned after vector search.
                Learn more about filtering syntax in [Metadata Filtering](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).
                Vector search happens first yielding `top_k` results, filtering is applied afterwards. Use
                `expand_top_k` parameter to increase amount of documents retrieved from `index` (`expand_top_k` take
                precedence if provided), filtering will make sure to return `top_k` out of `expand_top_k` documents
                ordered by score.
            top_k: How many documents to return.
            return_embedding: To return document embedding. By default is `None` which should reduce amount of data
                returned (considering embeddings are usually large in size)
            scale_score: Whether to scale the similarity score to the unit interval (range of [0,1]).
                If true (default) similarity scores (e.g. cosine or dot_product) which naturally have a different value
                range will be scaled to a range of [0,1], where 1 means extremely relevant. Otherwise raw similarity
                scores (e.g. cosine or dot_product) will be used.
            expand_top_k: The value will override `top_k` for vector search if provided. Should be used in case
                `filters` are expected to be applied on a greater amount of documents. After filtering takes place
                `top_k` documents retrieved ordered by score.

        Returns:
            Found `top_k` documents.
        """

        filter_ast = self._parse_filters(filters=filters)
        skip_properties = [] if return_embedding else [self.embedding_field]

        records = self.neo4j_client.query_embeddings(
            self.index, top_k, query_embedding, filter_ast, skip_properties, expand_top_k
        )
        results = [self._neo4j_record_to_document(rec) for rec in records]

        if scale_score:
            for document in results:
                document.score = self._scale_to_unit_interval(document.score)

        return results

    def delete_all_documents(
        self,
        document_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ):
        """
        Delete documents from the document store. All documents will be deleted, in case either `filters` or
        `document_ids` are defined only filtered subset will be deleted.

        Args:
            document_ids: Optional list of document ids to narrow down the documents to be deleted.
            filters: Optional filters to narrow down the documents which should be deleted.
                Learn more about filtering syntax in [Metadata Filtering](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).
        """

        filter_ast = self._parse_filters(ids=document_ids, filters=filters)
        self.neo4j_client.delete_nodes(self.node_label, filter_ast)

    def delete_index(self):
        """
        Deletes an existing index. The index including all data will be removed. The implementation deletes the index
        itself as well as all nodes having `self.node_label` label
        """

        try:
            self.neo4j_client.delete_index(self.index)
        except DatabaseError as err:
            if err.code == "Neo.DatabaseError.Schema.IndexDropFailed":
                logger.debug("Could not remove index `{index}`. Most probably it does not exist.")
            else:
                raise

        self.delete_all_documents()

    def create_index(self):
        self.neo4j_client.create_index(
            self.index, self.node_label, self.embedding_field, self.embedding_dim, self.similarity_function
        )

    def update_document_meta(self, document_id: str, meta: Dict[str, Any]):
        """
        Updates metadata properties in Neo4j for a Document found by its `document_id`. Please see details on how
        properties in nodes are being mutated in Neo4j for a given `meta` dictionary (https://neo4j.com/docs/cypher-manual/current/clauses/set/#set-setting-properties-using-map)

        Args:
            document_id: The Document id to update in Neo4j
            meta: Dictionary of new metadata. Will replace property values in case those already exist in the
                corresponding Neo4j node. Please notice it is assumed Document metadata has same schema (e.g. same
                amount of properties and its names) as in originally created nodes in Neo4j. Otherwise some side effects
                might be produced (e.g. a property is renamed in `meta` which leaves an old one in Neo4j).
        """

        updated_record = self.neo4j_client.update_node(self.node_label, document_id, meta)

        if not updated_record:
            logger.warning(
                "update_document_meta: Could not find document with id(%s) to update its meta attributes. "
                "Please check if provided id is valid",
                document_id,
            )

    def _handle_duplicate_documents(
        self,
        documents: List[Document],
        policy: DuplicatePolicy = None,
    ):
        """
        Checks whether any of the passed documents is already existing in the chosen index and returns a list of
        documents that are not in the index yet.

        :param documents: A list of Haystack Document objects.
        :param policy: Handle duplicates document based on parameter options.
                                    Parameter options : ( 'skip','overwrite','fail')
                                    skip (default option): Ignore the duplicates documents
                                    overwrite: Update any existing documents with the same ID when adding documents.
                                    fail: an error is raised if the document ID of the document being added already
                                    exists.
        :return: A list of `haystack.Document` objects.
        """

        if policy in (DuplicatePolicy.SKIP, DuplicatePolicy.FAIL):
            documents = self._drop_duplicate_documents(documents)
            documents_found = self.get_documents_by_id(document_ids=[doc.id for doc in documents])
            ids_exist_in_db: List[str] = [doc.id for doc in documents_found]

            if len(ids_exist_in_db) > 0 and policy == DuplicatePolicy.FAIL:
                msg = f"Document with ids '{', '.join(ids_exist_in_db)} already exists in index = '{self.index}'."
                raise DuplicateDocumentError(msg)

            documents = list(filter(lambda doc: doc.id not in ids_exist_in_db, documents))

        return documents

    def _drop_duplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Drop duplicates documents based on same hash ID

        :param documents: A list of Haystack Document objects.
        :param index: name of the index
        :return: A list of Haystack Document objects.
        """
        _hash_ids: Set = set()
        _documents: List[Document] = []

        for document in documents:
            if document.id in _hash_ids:
                logger.info(
                    "Duplicate Documents: Document with id '%s' already exists in index '%s'",
                    document.id,
                    self.index,
                )
                continue
            _documents.append(document)
            _hash_ids.add(document.id)

        return _documents

    def _get_distance(self, similarity: str) -> SimilarityFunction:
        """
        Validates similarity function so that it is supported by neo4j vector index.
        Only "cosine" and "l2" are supported aat the moment.

        Args:
            similarity: Common similarity values accepted by DocumentStores in Haystack,
                e.g. "cosine", "dot_product", "l2".

        Raises:
            ValueError: If given similarity is not supported by neo4j.

        Returns:
            Similarity function supported by Neo4j vector index ("cosine" or "euclidean").
        """
        try:
            return self.SIMILARITY_MAP[similarity]
        except KeyError as exc:
            raise ValueError(
                f"Provided similarity '{similarity}' is not supported by Neo4jDocumentStore. "
                f"Please choose one of the options: {', '.join(self.SIMILARITY_MAP.keys())}"
            ) from exc

    def _parse_filters(
        self,
        *,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[OperatorAST]:
        """
        Utility method which combines different filters in order to build a final one to be sent to `Neo4jClient`
        for execution. `FilterParser` will parse given `filters` as well as additional conditions (e.g. `ids`)
        and combine all those into a final syntax tree with `FilterParser.combine` (by default combines filters
        with `OP_AND` operator).

        Args:
            ids: Optional list of document ids to create a corresponding filter's ``IN`` expression,
                e.g. ``"ids IN ['id1', 'id2']"``
            filters: Filters to be parsed by `FilterParser.parse` in order to build a syntax tree.

        Returns:
            A syntax tree representing `filters` with additional conditions if any. `None` if none of conditions
            are defined.
        """
        ids_ast = self.filter_parser.comparison_op("id", COMPARISON_OPS.OP_IN, ids) if ids else None
        filter_ast = self.filter_parser.parse(filters) if filters else None

        return self.filter_parser.combine(ids_ast, filter_ast)

    def _neo4j_record_to_document(self, record: Neo4jRecord) -> Document:
        """
        Creates `Document` from Neo4j record (`dict`).
        """
        return Document.from_dict(record)

    def _document_to_neo4j_record(self, document: Document) -> Neo4jRecord:
        """
        Creates Neo4j record (`dict`) from `Document`. Please notice how `meta` fields stored on same
        level as `Document` fields. **It assumes attribute names (keys) do not clash**.
        """
        doc_object = document.to_dict(flatten=True)
        return doc_object

    def _scale_to_unit_interval(self, score: float) -> float:
        return (score + 1) / 2 if self.similarity == "cosine" else float(1 / (1 + np.exp(-score / 100)))

    def __del__(self):
        self.neo4j_client.close_driver()
