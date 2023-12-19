import logging
from typing import Any, ClassVar, Dict, Generator, List, Literal, Optional, Union

import numpy as np
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import DenseRetriever
from haystack.schema import Document, FilterType
from haystack.utils.batching import get_batches_from_generator
from neo4j.exceptions import DatabaseError
from tqdm import tqdm

from neo4j_haystack.document_stores.filters import (
    OP_EXISTS,
    OP_IN,
    FilterParser,
    OperatorAST,
)
from neo4j_haystack.document_stores.labels import Neo4jDocumentStoreLabels
from neo4j_haystack.document_stores.neo4j_client import (
    Neo4jClient,
    Neo4jClientConfig,
    Neo4jRecord,
)

logger = logging.getLogger(__name__)


SimilarityFunction = Literal["cosine", "euclidean"]


class Neo4jDocumentStore(Neo4jDocumentStoreLabels, BaseDocumentStore):
    """
    Document store for [Neo4j Database](https://neo4j.com/) with support for dense retrievals using \
    [Vector Search Index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/)

    The implementation is based on [Python Driver](https://neo4j.com/docs/python-manual/current/) for database access.
    Document properties are stored as graph nodes. Embeddings are stored as part of node properties along with the rest
    of attributes (including meta):

    ```py title="Document json representation (e.g. `Document.to_json`)"
    {
      "id": "793764",
      "id_hash_keys": ["content"],
      "content": "Aliens and UFOs are more real than ever before...",
      "content_type": "text",
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
        "id_hash_keys": [ "content" ],
        "content": "Aliens and UFOs are more real than ever before...",
        "content_type": "text",
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
    `embedding` but the actual dense retrieval is performed using a dedicated search index created automatically by
    `Neo4jDocumentStore`. The index is created using `db.index.vector.createNodeIndex()` Neo4j procedure and is based
    on the `embedding` property. Embedding dimension as well as similarity function (e.g. `cosine`) are configurable.

    At the moment Neo4j supports only cosine and euclidean(l2) similarity functions.

    Metadata filtering by `Neo4jDocumentStore` is performed using the standard `WHERE` Cypher query clause.
    Vector search is implemented by calling `db.index.vector.queryNodes()` procedure. **Neo4j currently does not support
    metadata "pre-filtering" which runs in combination with vector search. First vector search takes place and metadata
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
        embedding_field="embedding",
        index="document-embeddings", # The name of the Vector search index in Neo4j
        node_label="Document", # Providing a label to Neo4j nodes which store Documents
    )

    # Write documents to Neo4j. Respective nodes will be created. Here we assume no embeddings assigned (is null)
    document_store.write_documents(documents)

    # Retriever is using `document_store` to query documents
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        model_format="sentence_transformers",
    )

    # Create embeddings using `retriever` and store them in Neo4j (e.g. in the `embedding` property as configured above)
    document_store.update_embeddings(retriever)

    # Embed a query and hope to get an answer by querying Neo4j search index
    result: List[Document] = retriever.retrieve("Can I build GPT4 analog using my laptop and a lot of free time?")
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
        progress_bar: bool = True,
        duplicate_documents: str = "overwrite",
        create_index_if_missing: Optional[bool] = True,
        recreate_index: Optional[bool] = False,
    ):
        """Constructor method

        Args:
            url: URI pointing to Neo4j instance see (https://neo4j.com/docs/api/python-driver/current/api.html#uri)
            database: Neo4j database to interact with.
            username: Username to authenticate with the database.
            password: Password credential for the given username.
            client_config: Advanced client configuration to control various settings of underlying neo4j python
                driver. See :class:`Neo4jClientConfig` for more details. The mandatory `url` attribute will be set on
                the `client_config` in case it was provided in the config itself.
            index: The name of Neo4j Vector search used during index creation as well as querying embeddings
            node_label: The name of the label used in Neo4j to represent :class:`haystack.schema.Document`.
                Neo4j nodes are used primarily as storage for Document attributes and metadata filtering.
                The filtering process includes `node_label` in database queries (e.g.
                ``MATCH (doc:<node_label>) RETURN doc``). Together with the :attr:`.index` it identifies where
                documents are located in the database.
            embedding_dim: embedding dimension specified for the Vector search index.
            embedding_field: the name of embedding field which is created as a Neo4j node property containing an
                embedding vector. By default it is the same as in :class:`haystack.schema.Document`. It is used during
                index creation and querying embeddings. Please also notice how field name mappings between Neo4j nodes
                and Haystack Documents are being handled in the :meth:`._create_document_field_map`.
            similarity: similarity function specified during Vector search index creation. Supported values are
                "cosine" and "l2".
            progress_bar: Shows a tqdm progress bar.
            duplicate_documents: Handles duplicates document based on parameter options.
                Parameter options: ('skip','overwrite','fail')
                    - skip: Ignores the duplicate documents.
                    - overwrite: Updates any existing documents with the same ID when adding documents.
                    - fail: Raises an error if the document ID of the document being added already exists.
            create_index_if_missing: Will create vector index during class initialization if it is not yet available
                in the `database`. Will only take effect if `recreate_index` is not const:`True`.
            recreate_index: If `True` will delete existing index and its data (documents) and create a new
                index. Useful for testing purposes when a new DocumentStore initializes with a clean database state.

        Raises:
            ValueError: In case similarity function specified is not supported
        """

        super().__init__()

        self.index = index
        self.node_label = node_label
        self.embedding_dim = embedding_dim
        self.embedding_field = embedding_field

        self.similarity = similarity
        self.similarity_function = self._get_distance(similarity)

        self.progress_bar = progress_bar
        self.duplicate_documents = duplicate_documents

        self.filter_parser = FilterParser()

        if client_config and not client_config.url:
            client_config.url = url
        self.client_config = client_config or Neo4jClientConfig(url, database, username, password)
        self.neo4j_client = Neo4jClient(self.client_config)

        self.neo4j_client.verify_connectivity()

        if recreate_index:
            self.delete_index(index)
            self.create_index(index)
        elif create_index_if_missing:
            self.neo4j_client.create_index_if_missing(
                self.index, self.node_label, self.embedding_field, self.embedding_dim, self.similarity_function
            )

    def write_documents(
        self,
        documents: Union[List[dict], List[Document]],
        index: Optional[str] = None,
        batch_size: int = 1000,
        duplicate_documents: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Writes documents to the DocumentStore.

        Args:
            documents: List of `Dicts` or List of `Documents`. If they already contain the embeddings, we'll index
                them right away in Neo4j. If not, you can later call :meth:`update_embeddings` to create and index them.
            index: The index name for storing and searching Document embeddings. Should not be provided otherwise
                raises an exception. It is not being used in the implementation because documents are stored as
                neo4j nodes (specified by :attr:`.node_label`). Index is automatically updated by Neo4j in background on
                server side when embedding property is updated on nodes.
            batch_size: When working with large number of documents, batching can help reduce memory footprint.
            duplicate_documents: Handle duplicates document based on parameter options. Parameter options:
                - skip: Ignore the duplicates documents.
                - overwrite: Update any existing documents with the same ID when adding documents.
                - fail: an error is raised if the document ID of the document being added already exists
            headers: Is not supported by Neo4j driver. Raises exception if provided.

        Raises:
            DuplicateDocumentError: Exception triggers on duplicate document.
            NotImplementedError: Exception triggers if either `index` or `headers` are provided.
            ValueError: If `duplicate_documents` parameter is not one of `("skip", "overwrite", "fail")`
        """

        self._check_not_implemented(index, headers)

        if len(documents) == 0:
            logger.warning("Calling Neo4jDocumentStore.write_documents() with an empty list")
            return

        duplicate_documents = duplicate_documents or self.duplicate_documents
        if duplicate_documents not in self.duplicate_documents_options:
            raise ValueError(f"duplicate_documents parameter must be {', '.join(self.duplicate_documents_options)}")

        field_map = self._create_document_field_map()
        document_objects = [Document.from_dict(d, field_map=field_map) if isinstance(d, dict) else d for d in documents]
        document_objects = self._handle_duplicate_documents(
            documents=document_objects, index=self.index, duplicate_documents=duplicate_documents
        )

        batched_documents = get_batches_from_generator(document_objects, batch_size)
        with tqdm(
            total=len(document_objects),
            desc=f"Write Documents<index: {self.index},node_label: {self.node_label}>",
            unit=" docs",
            disable=not self.progress_bar,
        ) as progress_bar:
            for document_batch in batched_documents:
                records = [self._document_to_neo4j_record(doc, field_map) for doc in document_batch]
                embedding_field = self._map_document_field(self.embedding_field)
                self.neo4j_client.merge_nodes(self.node_label, embedding_field, records)
                progress_bar.update(batch_size)

    def update_embeddings(
        self,
        retriever: DenseRetriever,
        index: Optional[str] = None,
        update_existing_embeddings: bool = True,
        filters: Optional[FilterType] = None,
        batch_size: int = 32,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Updates the embeddings in the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever
        configuration).

        Args:
            retriever: Retriever to use to get embeddings for a `Document`
            index: The index name for storing and searching Document embeddings. Should not be provided otherwise
                throws an exception. It is not being used in the implementation because documents are being written to a
                Neo4j node (specified by :attr:`.node_label`). Index is automatically updated by Neo4j in background on
                server side when embedding property is updated on nodes.
            update_existing_embeddings: Whether to update existing embeddings of the documents. If set to
                `False`, only documents without embeddings are processed. This mode can be used for incremental
                updating of embeddings, wherein, only newly indexed documents get processed.
            filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                Example: `:::py3 {"category": ["one", "two"], "age": {"$eq": 20}}`. See more details in [Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
            batch_size: When working with large number of documents, batching can help reduce memory footprint.
            headers: Is not supported by Neo4j driver. Raises exception if provided.
        """
        only_documents_without_embedding = not update_existing_embeddings

        document_count = self.get_document_count(
            filters,
            index,
            only_documents_without_embedding,
        )
        if document_count == 0:
            logger.warning(
                "No documents found to `update_embeddings`. "
                "Consider checking `filters` and `update_existing_embeddings` parameters if that is not expected"
            )
            return

        logger.info("Updating embeddings for %s docs...", document_count)

        doc_generator = self.get_all_documents_generator(
            index=index,
            filters=filters,
            batch_size=batch_size,
            headers=headers,
            only_documents_without_embedding=self._true_or_none(only_documents_without_embedding),
        )

        with tqdm(
            total=document_count, disable=not self.progress_bar, unit=" docs", desc="Updating embeddings"
        ) as progress_bar:
            for document_batch in get_batches_from_generator(doc_generator, batch_size):
                embeddings = retriever.embed_documents(document_batch)
                self._validate_embeddings_shape(
                    embeddings=embeddings,
                    num_documents=len(document_batch),
                    embedding_dim=self.embedding_dim,
                )

                only_embeddings = [
                    {"id": doc.id, self.embedding_field: embedding}
                    for doc, embedding in zip(document_batch, embeddings)
                ]

                self.neo4j_client.update_embedding(
                    self.node_label,
                    self.embedding_field,
                    only_embeddings,
                )

                progress_bar.update(batch_size)

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 1000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        return list(self.get_all_documents_generator(index, filters, return_embedding, batch_size, headers))

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        return_embedding: Optional[bool] = None,
        batch_size: int = 1000,
        headers: Optional[Dict[str, str]] = None,
        only_documents_without_embedding: Optional[bool] = None,
    ) -> Generator[Document, None, None]:
        """
        Get all documents from the document store. Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory. Such mechanism is natively
        supported by underlying Neo4j Python Driver (an internal buffer which is depleted while being read and filled
        up while data is coming from the database)

        Args:
            index: Should not be provided otherwise throws an exception. Documents are retrieved from nodes, index
                is separate data structure used for vector search only.
            filters: Optional filters to narrow down the documents which should be returned.
                Example: ``{"category": ["one", "two"], "age": {"$eq": 20}}``. Learn more about filtering syntax in \
                [Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering)
            return_embedding: To return document embedding. By default is `None` which should reduce amount of data
                returned (considering embeddings are usually large in size)
            batch_size: When working with large number of documents, batching can help reduce memory footprint.
                This parameter controls how many documents are retrieved at once from Neo4j.
            headers: Is not supported by Neo4j driver. Raises exception if provided.
            only_documents_without_embedding: If set to `True`, only documents without embeddings are
                returned. `False` will return documents which do have embeddings. `None` will return either case (any
                document regardless of embedding presence).

        Raises:
            NotImplementedError: Exception triggers if either `index` or `headers` are provided.

        Returns:
            A Generator of found documents.
        """
        self._check_not_implemented(index, headers)

        filter_ast = self._parse_filters(filters=filters, documents_without_embedding=only_documents_without_embedding)
        skip_properties = [] if return_embedding else [self._map_document_field(self.embedding_field)]

        records = self.neo4j_client.find_nodes(self.node_label, filter_ast, skip_properties, fetch_size=batch_size)

        return (self._neo4j_record_to_document(rec) for rec in records)

    def get_document_by_id(
        self, id: str, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ) -> Optional[Document]:
        """
        Retrieves a document by its ID.

        Args:
            id: ID of the Document to retrieve.
            index: Should not be provided otherwise throws an exception. Documents are retrieved from nodes, index
                is separate data structure used for vector search only.
            headers: Is not supported by Neo4j driver. Raises exception if provided.

        Raises:
            NotImplementedError: Exception triggers if either `index` or `headers` are provided.

        Returns:
            A found document with matching `id` if exactly one is found, otherwise `None` is returned
        """
        self._check_not_implemented(index, headers)

        records = self.get_documents_by_id([id])
        number_found = len(records)

        if number_found > 1:
            logger.warn(
                f"get_document_by_id: Found more than one document for a given id(`{id}`). "
                "Expected: 1, Found: {number_found}. Please make sure your data has unique ids"
            )

        return records[0] if number_found > 0 else None

    def get_documents_by_id(
        self,
        ids: List[str],
        index: Optional[str] = None,
        batch_size: int = 1_000,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Document]:
        """
        Retrieves all documents using their IDs.

        Args:
            ids: List of IDs to retrieve.
            index: Should not be provided otherwise throws an exception. Documents are retrieved from nodes, index
                is separate data structure used for vector search.
            batch_size: Number of documents to retrieve at a time. When working with large number of documents,
                batching can help reduce memory footprint.
            headers: Is not supported by Neo4j driver. Raises exception if provided.

        Raises:
            NotImplementedError: Exception triggers if either `index` or `headers` are provided.

        Returns:
            List of found Documents.
        """
        self._check_not_implemented(index, headers)

        documents: List[Document] = []
        for batch_ids in get_batches_from_generator(ids, batch_size):
            filter_ast = self.filter_parser.comparison_op("id", OP_IN, list(batch_ids))
            records = self.neo4j_client.find_nodes(self.node_label, filter_ast)
            documents.extend([self._neo4j_record_to_document(rec) for rec in records])

        return documents

    def get_document_count(
        self,
        filters: Optional[FilterType] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Return the count of documents in the document store.

        Args:
            filters: Optional filters to narrow down the documents which should be counted.
                Example: {"category": ["one", "two"], "age": {"$eq": 20}}. See more details in \
                [Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering) for filter syntax
            index: Should not be provided otherwise throws an exception. Documents are counted based on nodes, index
                is separate data structure used for vector search only.
            only_documents_without_embedding: If set to `True`, only documents without embeddings are counted.
                Please notice it works in combination with filters if provided.
            headers: Is not supported by Neo4j driver. Raises exception if provided.

        Raises:
            NotImplementedError: Exception triggers if either `index` or `headers` are provided.

        Returns:
            Found documents count with respective filters applied.
        """
        self._check_not_implemented(index, headers)

        filter_ast = self._parse_filters(
            filters=filters, documents_without_embedding=self._true_or_none(only_documents_without_embedding)
        )

        return self.neo4j_client.count_nodes(self.node_label, filter_ast)

    def get_embedding_count(self, filters: Optional[FilterType] = None) -> int:
        """
        Return the number of embeddings in the document store.

        Args:
            filters: Optional filters to narrow down the documents which should be counted.
                Example: ``{"category": ["one", "two"], "age": {"$eq": 20}}``. See more details on filter syntax in \
                [Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering).
        """
        filter_ast = self._parse_filters(filters=filters, documents_without_embedding=False)

        return self.neo4j_client.count_nodes(self.node_label, filter_ast)

    def query_by_embedding(
        self,
        query_emb: np.ndarray,
        filters: Optional[FilterType] = None,
        top_k: int = 10,
        index: Optional[str] = None,
        return_embedding: Optional[bool] = None,
        headers: Optional[Dict[str, str]] = None,
        scale_score: bool = True,
        expand_top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.

        Args:
            query_emb: Embedding of the query (e.g. gathered from Dense Retrievers)
            filters: Optional filters to narrow down the documents which should be returned after vector search.
                Example: {"category": ["one", "two"], "age": {"$eq": 20}}. See more details in \
                [Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering) for filter syntax.
                Vector search happens first yielding `top_k` results, filtering is applied afterwards. Use
                `expand_top_k` parameter to increase amount of documents retrieved from `index` (`expand_top_k` take
                precedence if provided), filtering will make sure to return `top_k` out of `expand_top_k` documents
                ordered by score.
            top_k: How many documents to return.
            index: Index name to query vectors from. If `None`, the DocumentStore's default index
                name (:attr:`self.index`) will be used.
            return_embedding: To return document embedding. By default is `None` which should reduce amount of data
                returned (considering embeddings are usually large in size)
            headers: Is not supported by Neo4j driver. Raises exception if provided.
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
        self._check_not_implemented(headers=headers)
        index = index or self.index

        query_emb = query_emb.astype(np.float32)
        if self.similarity == "cosine":
            self.normalize_embedding(query_emb)

        filter_ast = self._parse_filters(filters=filters)
        skip_properties = [] if return_embedding else [self.embedding_field]

        records = self.neo4j_client.query_embeddings(
            index, top_k, query_emb.tolist(), filter_ast, skip_properties, expand_top_k
        )
        results = [self._neo4j_record_to_document(rec) for rec in records]

        if scale_score:
            for document in results:
                document.score = self.scale_to_unit_interval(document.score, self.similarity)

        return results

    def delete_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete all documents from the document store. This method is deprecated. Use :meth:`delete_documents` instead.
        """
        logger.warning(
            """DEPRECATION WARNINGS:
                1. delete_all_documents() method is deprecated, please use delete_documents method
                For more details, please refer to the issue: https://github.com/deepset-ai/haystack/issues/1045
                """
        )
        self.delete_documents(index, None, filters, headers)

    def delete_documents(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Delete documents from the document store. All documents will be deleted, in case either`filters` or `ids`
        are defined only filtered subset will be deleted.

        Args:
            index: The index name for storing and searching Document embeddings. Should not be provided otherwise
                raises an exception. It is not being used in the implementation because documents are stored as
                Neo4j nodes (specified by :attr:`.node_label`). Index is automatically updated by Neo4j in background on
                server side when embedding property is updated on nodes.
            ids: Optional list of document IDs to narrow down the documents to be deleted.
            filters: Optional filters to narrow down the documents which should be counted.
                Example: `{"category": ["one", "two"], "age": {"$eq": 20}}`. See more details in \
                [Metadata Filtering](https://docs.haystack.deepset.ai/docs/metadata-filtering) for filter syntax
            headers: Is not supported by Neo4j driver. Raises exception if provided.

        Raises:
            NotImplementedError: Exception triggers if either `index` or `headers` are provided.
        """
        self._check_not_implemented(index, headers)
        filter_ast = self._parse_filters(ids=ids, filters=filters)
        self.neo4j_client.delete_nodes(self.node_label, filter_ast)

    def delete_index(self, index: Optional[str] = None):
        """
        Deletes an existing index. The index including all data will be removed. The implementation deletes the index
        itself as well as all nodes having :attr:`self.node_label` label

        Args:
            index: The name of the index to delete. If `None`, the DocumentStore's default index (:attr:`self.index`)
                will be used.
        """
        index = index or self.index

        try:
            self.neo4j_client.delete_index(index)
        except DatabaseError as err:
            if err.code == "Neo.DatabaseError.Schema.IndexDropFailed":
                logger.debug("Could not remove index `{index}`. Most probably it does not exist.")
            else:
                raise

        self.delete_documents()

    def create_index(self, index: Optional[str] = None):
        index = index or self.index
        self.neo4j_client.create_index(
            self.index, self.node_label, self.embedding_field, self.embedding_dim, self.similarity_function
        )

    def update_document_meta(self, id: str, meta: Dict[str, Any], index: Optional[str] = None):
        """
        Updates metadata properties in Neo4j for a Document found by its `id`. Please see details on how properties in
        nodes are being mutated in Neo4j for a given `meta` dictionary (https://neo4j.com/docs/cypher-manual/current/clauses/set/#set-setting-properties-using-map)

        Args:
            id: The Document id to update in Neo4j
            meta: Dictionary of new metadata. Will replace property values in case those already exist in the
                corresponding Neo4j node. Please notice it is assumed Document metadata has same schema (e.g. same
                amount of properties and its names) as in originally created nodes in Neo4j. Otherwise some side effects
                might be produced (e.g. a property is renamed in `meta` which leaves an old one in Neo4j).
            index: The index name for storing and searching Document embeddings. Should not be provided otherwise
                raises an exception. It is not being used in the implementation because documents are stored as
                Neo4j nodes (specified by :attr:`.node_label`). Index is automatically updated by Neo4j in background
                when embedding property is updated on nodes.
        """
        self._check_not_implemented(index)

        updated_record = self.neo4j_client.update_node(self.node_label, id, meta)

        if not updated_record:
            logger.warning(
                "update_document_meta: Could not find document with id(%s) to update its meta attributes. "
                "Please check if provided id is valid",
                id,
            )

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
            Similarity function supported by neo4j vector index ("cosine", "euclidean").
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
        filters: Optional[FilterType] = None,
        documents_without_embedding: Optional[bool] = None,
    ) -> Optional[OperatorAST]:
        """
        Utility method which combines different filters in order to build a final one to be sent to :class:`Neo4jClient`
        for execution. :class:`FilterParser` will parse given `filters` as well as additional conditions (e.g. `ids`)
        and combine all those into a final syntax tree with :py:meth:`FilterParser.combine` (by default combines filters
        with `OP_AND` operator).

        Args:
            ids: Optional list of document ids to create a corresponding filter's ``IN`` expression,
                e.g. ``"ids IN ['id1', 'id2']"``
            filters: Filters to be parsed by :meth:`FilterParser.parse` in order to build a syntax tree.
            documents_without_embedding: If `True` or `False` creates additional filter expression
                (e.g. ``{ "embedding": { "$exists": False } }``) using `OP_EXISTS` operator to return documents
                without or with embeddings respectively. As metadata filtering is not strictly defined `OP_EXISTS`
                operator was introduced for convenience. See a similar "$exists" operator in [MongoDB](https://www.mongodb.com/docs/manual/reference/operator/query/exists/)

        Returns:
            A syntax tree representing `filters` with additional conditions if any. `None` if none of conditions
            are defined.
        """
        ids_ast = self.filter_parser.comparison_op("id", OP_IN, ids) if ids else None
        filter_ast = self.filter_parser.parse(filters) if filters else None
        embedding_exists_ast = (
            self.filter_parser.comparison_op(self.embedding_field, OP_EXISTS, not documents_without_embedding)
            if documents_without_embedding is not None
            else None
        )

        return self.filter_parser.combine(ids_ast, embedding_exists_ast, filter_ast)

    def _create_document_field_map(self) -> Dict:
        return {self.embedding_field: "embedding"}

    def _map_document_field(self, field_name: str) -> str:
        return self._create_document_field_map().get(field_name, field_name)

    def _neo4j_record_to_document(self, record: Neo4jRecord, field_map: Optional[Dict[str, Any]] = None) -> Document:
        """
        Creates :class:`Document` from Neo4j record (a :class:`dict`).
        """
        field_map = field_map or self._create_document_field_map()
        return Document.from_dict(record, field_map)

    def _document_to_neo4j_record(self, document: Document, field_map: Optional[Dict[str, Any]] = None) -> Neo4jRecord:
        """
        Creates neo4j record (a :class:`dict`) from :class:`Document`. Please notice how meta fields stored on same
        level as :class:`Document` fields. It assumes attribute names (keys) do not clash.
        """
        field_map = field_map or self._create_document_field_map()
        doc_object = document.to_dict(field_map)
        doc_meta = doc_object.pop("meta")

        return {**doc_object, **doc_meta}

    def _check_not_implemented(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None):
        """
        Ensures both `index` and `headers` are `None`. If `headers` is defined raises :exc:`NotImplementedError`.
        Not throwing error when `index` is not None because :class:`BaseDocumentStore` might have logic which re-assigns
        its value, instead  warning is shown.
        """
        if index:
            # Not raising an error as BaseDocumentStore might have logic which assigns `index` a value
            logger.warning(
                "Neo4jDocumentStore does not support `index` parameter. "
                "Instead it uses `node_label` to refer to neo4j nodes to query and store Documents. "
                "`node_label` as well as `index` parameters can be provided in `Neo4jDocumentStore` constructor. "
                "It uses those parameters to create vector index. "
                "See https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/ for more details."
            )
        if headers:
            raise NotImplementedError("Neo4jDocumentStore does not support `headers`.")

    def _true_or_none(self, flag: bool) -> Optional[bool]:
        return True if flag else None

    def __del__(self):
        self.neo4j_client.close_driver()
