import logging
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    cast,
)

from haystack import default_from_dict, default_to_dict
from neo4j import (
    Auth,
    GraphDatabase,
    ManagedTransaction,
    Record,
    Result,
    Session,
    unit_of_work,
)

from neo4j_haystack.errors import Neo4jClientError
from neo4j_haystack.metadata_filter import AST, Neo4jQueryConverter

logger = logging.getLogger(__name__)

NODE_VAR = "doc"
"""Default variable name used in Cypher queries to match and return Documents, e.g.
`:::cypher match(doc:Document) where doc.id = $id return doc` where `doc` is a variable name."""

Neo4jRecord = Dict[str, Any]
"""Type alias for data items returned from Neo4j queries"""

SimilarityFunction = Literal["cosine", "euclidean"]

Neo4jSessionConfig = Mapping[str, Any]
"""Generic dictionary for [Session Configuration](https://neo4j.com/docs/api/python-driver/current/api.html#session-configuration)"""

Neo4jDriverConfig = Mapping[str, Any]
"""Generic dictionary for [Driver Configuration](https://neo4j.com/docs/api/python-driver/current/api.html#driver-configuration)"""

Neo4jTransactionConfig = Mapping[str, Any]
"""Generic dictionary for [Transaction Configuration](https://neo4j.com/docs/api/python-driver/current/api.html#transaction)"""


@dataclass
class VectorStoreIndexInfo:
    """Neo4j vector index information retrieved from the database.

    See [Create and configure vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-create)
    documentation to learn more about data representing index configuration.

    Attributes:
        index_name: The name of the index.
        node_label: Name of Neo4j node which contains embeddings which are indexed.
        property_key: Name of the property of the node which contains vectors.
        dimensions: Dimension of embedding vector.
        similarity_function: Configured vector similarity function.
    """

    index_name: str
    node_label: str
    property_key: str
    dimensions: int
    similarity_function: str


@dataclass
class Neo4jClientConfig:
    """
    Provides extensive configuration options in order to communicate with Neo4j database.

    It combines several configuration levels for each entity used by python driver to communicate with a database:

    - [Driver Configuration][neo4j_haystack.client.neo4j_client.Neo4jDriverConfig]
    - [Session Configuration][neo4j_haystack.client.neo4j_client.Neo4jSessionConfig]
    - [Transaction Configuration][neo4j_haystack.client.neo4j_client.Neo4jTransactionConfig]

    Developers can pick up configuration properties for each entity (e.g. session) which will be applied during
    transaction invocations. For example, ``driver_config={"connection_timeout": 30}`` will set amount of time in
    seconds to wait for a TCP connection to be established.

    `username` and `password` are optional because developer can choose to provide alternative
    authentication options using `driver_config` by setting [Driver Auth Details](https://neo4j.com/docs/api/python-driver/current/api.html#auth).

    Attributes:
        url: Database connection string, see https://neo4j.com/docs/api/python-driver/current/api.html#uri.
        database: Database name to connect.
        username: Username to authenticate with the database.
        password: Password credential for the given username.
        driver_config: Additional driver configuration.
        session_config: Additional session configuration.
        transaction_config: Additional transaction configuration (e.g. ``timeout``)

    Raises:
        ValueError: In case conflicting auth credentials are provided - choose either username/password combination
            or `driver_config.auth`.
    """

    url: Optional[str] = field(default="bolt://localhost:7687")
    database: Optional[str] = field(default="neo4j")
    username: Optional[str] = field(default="neo4j")
    password: Optional[str] = field(default="neo4j")

    driver_config: Neo4jDriverConfig = field(default_factory=dict)
    session_config: Neo4jSessionConfig = field(default_factory=dict)
    transaction_config: Neo4jTransactionConfig = field(default_factory=dict)

    use_env: Optional[bool] = field(default=False)
    auth: Optional[Auth] = field(default=None)

    def __post_init__(self):
        if self.use_env:
            self.url = os.getenv("NEO4J_URL", self.url)
            self.database = os.getenv("NEO4J_DATABASE", self.database)
            self.username = os.getenv("NEO4J_USERNAME", self.username)
            self.password = os.getenv("NEO4J_PASSWORD", self.password)

        if self.username and self.password:
            self.auth = (self.username, self.password)

        if not self.url:
            raise ValueError("The `url` attribute is mandatory to connect to database.")

        if not self.auth:
            raise ValueError("Please provide either (`username`, `password`) or `auth` fields for authentication.")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes client configuration to a dictionary.
        """
        data = default_to_dict(
            self,
            url=self.url,
            database=self.database,
            username=self.username,
            password=self.password,
            driver_config=self.driver_config,
            session_config=self.session_config,
            transaction_config=self.transaction_config,
            use_env=self.use_env,
        )

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neo4jClientConfig":
        """
        Deserializes client configuration from a dictionary.
        """
        return default_from_dict(cls, data)


class Neo4jClient:
    """
    Neo4j Python Driver wrapper to run low level database transactions using Cypher queries. It abstracts away Neo4j
    related details from `Neo4jDocumentStore` so that database related interactions are encapsulated in a single
    place.

    `Neo4jClient` can be created with a number of configuration options represented by the `Neo4jClientConfig` data
    class. The configuration applied when connecting to a database or running transactions.

    Attributes:
        _config: Neo4j configuration options.
        _driver: An instance of [neo4j.Driver][] which is used to start a session for transaction execution.
        _filter_converter: Instance of `Neo4jQueryConverter` which converts parsed Metadata filters to Cypher
            queries.
    """

    def __init__(self, config: Neo4jClientConfig):
        self._config = config

        if not config.url:
            raise ValueError("`Neo4jClientConfig.url` is mandatory attribute when trying to connect to Neo4j database.")

        self._driver = GraphDatabase.driver(config.url, auth=config.auth, **config.driver_config)
        self._filter_converter = Neo4jQueryConverter(NODE_VAR)

    def delete_nodes(self, node_label: str, filter_ast: Optional[AST] = None) -> None:
        """
        Deletes nodes with with given label and filters using [DELETE](https://neo4j.com/docs/cypher-manual/current/clauses/delete/)
            Cypher clause.

        Args:
            node_label: The name of the label to delete (e.g. ``"Document"``)
            filter_ast: Metadata filters to delete only specific nodes which match filtering conditions.
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction) -> None:
            where_clause, where_params = self._where_clause(filter_ast)
            tx.run(
                f"""
                MATCH ({NODE_VAR}:`{node_label}`)
                {where_clause}
                DETACH DELETE {NODE_VAR}
                """,
                parameters={**where_params},
            )

        with self._begin_session() as session:
            session.execute_write(_mgt_tx)

    def create_index(
        self,
        index_name: str,
        label: str,
        property_key: str,
        dimension: int,
        similarity_function: SimilarityFunction,
    ) -> None:
        """
        Creates a new vector index in database for a given node label and vector specific attributes (e.g. dimension,
        similarity function etc). See documentation for the index creation procedure \
        [db.index.vector.createNodeIndex](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_createNodeIndex)

        Args:
            index_name: The unique name of the index.
            label: The node label to be indexed (e.g. ``"Document"``).
            property_key: The property key of a node which contains embedding values.
            dimension: Vector embedding dimension (must be between 1 and 2048 inclusively).
            similarity_function: case-insensitive values for the vector similarity function:
                ``cosine`` or ``euclidean``.
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction) -> None:
            tx.run(
                """
                CALL db.index.vector.createNodeIndex(
                    $index_name,
                    $label,
                    $property_key,
                    toInteger($vector_dimension),
                    $similarity_function
                )
                """,
                index_name=index_name,
                label=label,
                property_key=property_key,
                vector_dimension=dimension,
                similarity_function=similarity_function,
            )

        with self._begin_session() as session:
            session.execute_write(_mgt_tx)

    def retrieve_vector_index(
        self,
        index_name: str,
        node_label: str,
        property_key: str,
    ) -> Optional[VectorStoreIndexInfo]:
        """
        Retrieves information about existing vector index.

        For more details and an example query on how to obtain existing indexes see \
        [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query).

        Args:
            index_name: The name of the vector index to retrieve.
            node_label: The label of the node configured as prt of vector index setup.
            property_key: The property key configured as part of vector index setup.

        Raises:
            Neo4jClientError: If more than one index found matching search criteria (same index name OR
                label+property combination).

        Returns:
            Data retrieved from the query execution or `None` if index was not found.
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction) -> List[Record]:
            result = tx.run(
                """
                SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
                WHERE type = 'VECTOR' AND
                (name = $index_name OR (labelsOrTypes[0] = $node_label AND properties[0] = $property_key))
                RETURN name, labelsOrTypes, properties, options
                """,
                index_name=index_name,
                node_label=node_label,
                property_key=property_key,
            )

            return list(result)

        with self._begin_session() as session:
            found_indexes = session.execute_write(_mgt_tx)

        if len(found_indexes) > 1:
            raise Neo4jClientError(
                "Failed to retrieve vector index from Neo4j."
                "There were several indexes found with a given search criteria: "
                f"$index_name='{index_name}' OR ($node_label='{node_label}' AND $property_key='{property_key}'). "
                "Please make sure the Neo4jDocumentStore points to an unambiguous vector index"
            )

        return self._vector_store_index_info(found_indexes[0]) if found_indexes else None

    def create_index_if_missing(
        self,
        index_name: str,
        label: str,
        property_key: str,
        dimension: int,
        similarity_function: SimilarityFunction,
    ):
        """
        Creates a vector index in case it does not exist in database.
        Uses same parameters as [create_index][neo4j_haystack.client.neo4j_client.Neo4jClient.create_index] \
            method.
        """

        existing_index = self.retrieve_vector_index(index_name, label, property_key)

        if not existing_index:
            logger.debug("Creating a new index(%s) as it is not present in the configured Neo4j database", index_name)
            self.create_index(index_name, label, property_key, dimension, similarity_function)

    def delete_index(self, index_name: str) -> None:
        """
        Removes index from Neo4j database.

        See Cypher manual on [Drop vector indexes](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-drop)

        Args:
            index_name: The name of the index to delete.
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction) -> None:
            tx.run(f"DROP INDEX `{index_name}`")

        with self._begin_session() as session:
            session.execute_write(_mgt_tx)

    def update_embedding(self, node_label: str, embedding_field: str, records: List[Dict[str, Any]]) -> None:
        """
        Updates embedding on a number of ``Document`` nodes. It uses ``db.create.setNodeVectorProperty()`` procedure as
        a recommended update method. See more details in [Set a vector property on a node](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-set)

        Args:
            node_label: A node label to match (e.g. ``"Document"``).
            embedding_field: The name of the embedding field which stores embeddings (of type ``LIST<FLOAT>``) as part
                node properties.
            records: A list dictionary objects following the structure:
                ```python
                    [{
                        "id": "doc_id1", # id of the Document (node) to update
                        embedding_field: [0.8, 0.9, ...] # Embedding vector
                    }]
                ```
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction) -> None:
            tx.run(
                f"""
                WITH $records AS batch
                UNWIND batch as row
                MATCH ({NODE_VAR}:`{node_label}` {{id: row.id}})
                CALL db.create.setNodeVectorProperty({NODE_VAR}, '{embedding_field}', row.{embedding_field})
                RETURN {NODE_VAR}
                """,
                records=records,
            )

        with self._begin_session() as session:
            session.execute_write(_mgt_tx)

    def merge_nodes(self, node_label: str, embedding_field: str, records: List[Neo4jRecord]) -> None:
        """
        Creates or updates a node in neo4j representing a Document with all properties. Nodes are matched by "id",
        if not found a new node will be created. See the following manuals:

        - [MERGE clause](https://neo4j.com/docs/cypher-manual/current/clauses/merge/)
        - [Settings properties using a map](https://neo4j.com/docs/cypher-manual/current/clauses/set/#set-setting-properties-using-map)
        - [db.create.setNodeVectorProperty](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_create_setNodeVectorProperty)

        Args:
            node_label: The label of the node to match (e.g. "Document").
            embedding_field: The name of the embedding field which stores embeddings (of type ``LIST<FLOAT>``) as part
                of node properties. Embeddings (if present) will be updated/set by ``db.create.setNodeVectorProperty()``
                procedure - `embedding_field` is excluded from ``SET`` Cypher clause by using map projections.
            records: A list of [Documents](https://docs.haystack.deepset.ai/reference/primitives-api#document) \
                converted to dictionaries, with ``meta`` attributes included.
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction):
            tx.run(
                f"""
                WITH $records AS batch
                UNWIND batch as row
                MERGE ({NODE_VAR}:`{node_label}` {{id: row.id}})
                SET {NODE_VAR} += row{{.*, {embedding_field}: null}}
                WITH {NODE_VAR}, row
                CALL {{ WITH {NODE_VAR}, row
                    MATCH({NODE_VAR}:`{node_label}` {{id: row.id}}) WHERE row.embedding IS NOT NULL
                    CALL db.create.setNodeVectorProperty({NODE_VAR}, '{embedding_field}', row.{embedding_field})
                }}
                """,
                records=records,
            )

        with self._begin_session() as session:
            session.execute_write(_mgt_tx)

    def count_nodes(self, node_label: str, filter_ast: Optional[AST] = None) -> int:
        """
        Counts number of nodes matching given label and optional filters.

        Args:
            node_label: The label of the node to match (e.g. ``"Document"``).
            filter_ast: The filter syntax tree (parsed metadata filter) to narrow down counted results.

        Returns:
            Number of found nodes.
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction):
            where_clause, where_params = self._where_clause(filter_ast)
            result = tx.run(
                f"""
                MATCH ({NODE_VAR}:`{node_label}`)
                {where_clause}
                RETURN count(*) as count
                """,
                parameters={**where_params},
            )
            return result.single(strict=True).value()

        with self._begin_session() as session:
            return session.execute_read(_mgt_tx)

    def find_nodes(
        self,
        node_label: str,
        filter_ast: Optional[AST] = None,
        skip_properties: Optional[List[str]] = None,
        fetch_size: int = 1000,
    ) -> Generator[Neo4jRecord, None, None]:
        """
        Search for nodes matching a given label and metadata filters.

        Args:
            node_label: The label of the nodes to match (e.g. ``"Document"``).
            filter_ast: The filter syntax tree (parsed metadata filter) for search.
            skip_properties: Properties we would like not to return as part of data payload. Is uses map projection
                Cypher syntax, e.g. `:::cypher doc{.*, embedding: null}` - such construct will make sure ``embedding``
                is not returned back in results.
            fetch_size: Controls how many records are fetched at once from the database which helps with batching
                process.

        Returns:
            Found records matching search criteria.
        """
        where_clause, where_params = self._where_clause(filter_ast)
        query = f"""
            MATCH ({NODE_VAR}:`{node_label}`)
            {where_clause}
            RETURN {NODE_VAR}{self._map_projection(skip_properties)}
            """

        for record in self.query_nodes(query=query, parameters={**where_params}, fetch_size=fetch_size):
            yield cast(Neo4jRecord, record.data().get(NODE_VAR))

    def query_nodes(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        fetch_size: int = 1000,
    ) -> Generator[Record, None, None]:
        """
        Runs a given Cypher `query`. The implementation is based on ``Unmanaged Transactions``
        for greater control and possibility to ``yield`` results as soon as those are fetched from database. The Neo4j
        python driver internally manages a buffer which replenished while records are being consumed thus making sure we
        do not store all fetched records in memory. That greatly simplifies batching mechanism as it is implemented by
        the buffer. See more details about how python driver implements \
        [Explicit/Unmanaged Transactions](https://neo4j.com/docs/api/python-driver/current/api.html#explicit-transactions-unmanaged-transactions)

        Note:
            Please notice results are yielded while read transaction is still open. That should impact your choice of
            transaction timeout setting, see \
                [Neo4jClientConfig][neo4j_haystack.client.neo4j_client.Neo4jClientConfig].

        Args:
            query: Cypher query to run in Neo4j.
            parameters: Query parameters which can be used as placeholders in the `query`.
            fetch_size: Controls how many records are fetched at once from the database which helps with batching
                process.

        Returns:
            Records containing data specified in ``RETURN`` Cypher query statement.
        """
        with self._begin_session(fetch_size=fetch_size) as session:
            with session.begin_transaction(
                metadata=self._config.transaction_config.get("metadata"),
                timeout=self._config.transaction_config.get("timeout"),
            ) as tx:
                try:
                    result: Result = tx.run(
                        query,
                        parameters=parameters,
                    )
                    yield from result
                finally:
                    tx.close()

    def query_embeddings(
        self,
        index: str,
        top_k: int,
        embedding: List[float],
        filter_ast: Optional[AST] = None,
        skip_properties: Optional[List[str]] = None,
        vector_top_k: Optional[int] = None,
    ) -> List[Neo4jRecord]:
        """
        Query a vector index and apply filtering using `WHERE` clause on results returned by vector search.
        See the following documentation for more details:

        - [Query a vector index](https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query)
        - [db.index.vector.queryNodes()](https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes)

        Args:
            index: Refers to the unique name of the vector index to query.
            top_k: Number of results to return from vector search.
            embedding: The query vector (a ``LIST<FLOAT>``) in which to search for the neighborhood.
            filter_ast: Additional filters translated into `WHERE` Cypher clause by \
                [Neo4jQueryConverter][neo4j_haystack.metadata_filter.Neo4jQueryConverter]
            skip_properties: Properties we would like **not** to return as part of data payload. Is uses map projection
                Cypher syntax, e.g. `:::cypher doc{.*, embedding: null}` - such construct will make sure `embedding` is
                not returned back in results.
            vector_top_k: If provided `vector_top_k` is used instead of `top_k` in order to increase number of
                results (nearest neighbors) from vector search. It makes sense when filters (`filter_ast`) could
                further narrow down vector search result. Only `top_k` number of records will be returned back thus
                `vector_top_k` should be preferably greater than `top_k`.
        Returns:
            An ordered by score `top_k` nodes found in vector search which are optionally filtered using
                ``WHERE`` clause.
        """

        score_property = "score"

        if vector_top_k and vector_top_k < top_k:
            logger.warning(
                "Make sure 'vector_top_k'(=%s) is greater than 'top_k'(=%s) parameter. Using 'top_k' instead",
                vector_top_k,
                top_k,
            )
            vector_top_k = top_k

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction) -> List[Record]:
            where_clause, where_params = self._where_clause(filter_ast)
            result = tx.run(
                f"""
                CALL db.index.vector.queryNodes($index, $vector_top_k, $embedding)
                YIELD node as {NODE_VAR}, {score_property}
                MATCH ({NODE_VAR}) {where_clause}
                RETURN {NODE_VAR}{self._map_projection(skip_properties)}, {score_property}
                ORDER BY {score_property} DESC LIMIT $top_k
                """,
                parameters={
                    "index": index,
                    "top_k": top_k,
                    "embedding": embedding,
                    "vector_top_k": vector_top_k or top_k,
                    **where_params,
                },
            )
            return list(result)

        with self._begin_session() as session:
            records = session.execute_read(_mgt_tx)

        return [{**record.value(NODE_VAR), score_property: record.value("score")} for record in records]

    def update_node(self, node_label: str, doc_id: str, data: Dict[str, Any]) -> Optional[Neo4jRecord]:
        """
        Updates a given node matched by the given id (`doc_id`). Properties are mutated by `+=` operator,
        see more details in [Setting properties using map](https://neo4j.com/docs/cypher-manual/current/clauses/set/#set-setting-properties-using-map).

        Args:
            node_label: A node label to match (e.g. "Document").
            doc_id: Node id to match. Please notice the `id` used in Cypher query is not a native element id but
                the one which mapped from the [haystack.schema.Document](https://docs.haystack.deepset.ai/reference/primitives-api#document).
            data: A dictionary of data which will be set as node's properties.

        Returns:
            Updated Neo4j record data.
        """

        @self._unit_of_work()
        def _mgt_tx(tx: ManagedTransaction):
            result = tx.run(
                f"""
                MATCH ({NODE_VAR}:`{node_label}` {{id: $doc_id}})
                SET {NODE_VAR} += $doc_data
                RETURN {NODE_VAR}
                """,
                doc_id=doc_id,
                doc_data=data,
            )
            return result.single(strict=False)

        with self._begin_session() as session:
            record = session.execute_write(_mgt_tx)

        return record.data().get(NODE_VAR) if record else None

    def verify_connectivity(self):
        """
        Verifies connection to Neo4j database as per configuration and auth credentials provided.

        Raises:
            Neo4jClientError: In case connection could not be established.
        """
        try:
            self._driver.verify_connectivity()
        except Exception as err:
            raise Neo4jClientError(
                "Could not connect to Neo4j database. Please ensure that the url and provided credentials are correct"
            ) from err

    def close_driver(self) -> None:
        logger.debug("Closing driver instance created for Neo4j client to release its connection pool")
        self._driver.close()

    def _begin_session(self, **session_kwargs) -> Session:
        """
        Creates a database session with common as well as client specific configuration settings.

        Returns:
            A new `Session` object to execute transactions.
        """
        session_config = {**self._config.session_config, **session_kwargs}
        return self._driver.session(database=self._config.database, **session_config)

    def _unit_of_work(self) -> Callable:
        """
        An extended version of managed transaction decorator to pass through configuration options from
        `self._config.transaction_config`:

        - ``metadata`` - will be attached to the executing transaction
        - ``timeout`` - the transaction timeout in seconds

        See more details in [Managed Transactions](https://neo4j.com/docs/api/python-driver/current/api.html#managed-transactions-transaction-functions)

        Returns:
            A pre-configured [neo4j.unit_of_work][] decorator
        """
        return unit_of_work(
            metadata=self._config.transaction_config.get("metadata"),
            timeout=self._config.transaction_config.get("timeout"),
        )

    def _where_clause(self, filter_ast: Optional[AST]) -> Tuple[str, Dict[str, Any]]:
        """
        Converts a given filter syntax tree `filter_ast` into a Cypher query in order to build ``WHERE`` filter clause.
        Along with the query method also returns parameters used in the query to be included into final request.
        Find out more details about [WHERE clause](https://neo4j.com/docs/cypher-manual/current/clauses/where/)

        Args:
            filter_ast: Filters AST to be converted into Cypher query by \
                [Neo4jQueryConverter.convert][neo4j_haystack.metadata_filter.Neo4jQueryConverter.convert].
        Returns:
            ``WHERE`` filter clause used in filtering logic (e.g. `:::cypher WHERE doc.age > $age`) as well as
            parameters used in the clause  (e.g. `:::py {"age": 25}`)
        """
        if filter_ast:
            query, params = self._filter_converter.convert(filter_ast)
            return f"WHERE {query}", params

        # empty query and no parameters
        return ("", {})

    def _map_projection(self, skip_properties: Optional[List[str]]) -> str:
        """
        Creates a map projection Cypher query syntax with the option to skip certain properties.
        Example query would be `:::cypher {.*, embedding=null}`, where `:::py skip_properties=["embedding"]`

        See Neo4j manual about [Map Projections](https://neo4j.com/docs/cypher-manual/current/values-and-types/maps/#cypher-map-projection)

        Args:
            skip_properties: a list of property names to skip (set values to ``null``) from map projection.

        Returns:
            A map projection Cypher query with skipped properties if any.
        """
        all_props = [".*"] + ([f"{p}: null" for p in skip_properties] if skip_properties else [])
        return f"{{{','.join(all_props)}}}"

    def _vector_store_index_info(self, record: Record) -> VectorStoreIndexInfo:
        """
        Creates a dataclass from a data record returned by a ``SHOW INDEXES`` Cypher query output.

        See Neo4j manual for [SHOW INDEXES](https://neo4j.com/docs/cypher-manual/current/indexes-for-search-performance/#indexes-list-indexes)

        Args:
            record: A Neo4j record containing ``SHOW INDEXES`` output.

        Returns:
            Custom dataclass with vector index information.
        """
        return VectorStoreIndexInfo(
            index_name=record["name"],
            node_label=record["labelsOrTypes"][0],
            property_key=record["properties"][0],
            dimensions=record["options"]["indexConfig"]["vector.dimensions"],
            similarity_function=record["options"]["indexConfig"]["vector.similarity_function"],
        )
