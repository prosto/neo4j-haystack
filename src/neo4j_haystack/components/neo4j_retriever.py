from typing import Any, Dict, List, Optional, cast

from haystack import (
    ComponentError,
    Document,
    component,
    default_from_dict,
    default_to_dict,
)

from neo4j_haystack.client import Neo4jClient, Neo4jClientConfig
from neo4j_haystack.document_stores import Neo4jDocumentStore


@component
class Neo4jEmbeddingRetriever:
    """
    A component for retrieving documents from Neo4jDocumentStore.

    ```py title="Retrieving documents assuming documents have been previously indexed"
    from haystack import Document, Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder

    from neo4j_haystack import Neo4jDocumentStore, Neo4jEmbeddingRetriever

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Document store with default credentials
    document_store = Neo4jDocumentStore(
        url="bolt://localhost:7687",
        embedding_dim=384, # same as the embedding model
    )

    pipeline = Pipeline()
    pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder(model=model_name))
    pipeline.add_component("retriever", Neo4jEmbeddingRetriever(document_store=document_store))
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

    result = pipeline.run(
        data={
            "text_embedder": {"text": "Query to be embedded"},
            "retriever": {
                "top_k": 5,
                "filters": {"field": "release_date", "operator": "==", "value": "2018-12-09"},
            },
        }
    )

    # Obtain retrieved documents from pipeline execution
    documents: List[Document] = result["retriever"]["documents"]
    ```
    """

    def __init__(
        self,
        document_store: Neo4jDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        scale_score: bool = True,
        return_embedding: bool = False,
    ):
        """
        Create a Neo4jEmbeddingRetriever component.

        Args:
            document_store: An instance of `Neo4jDocumentStore`.
            filters: A dictionary with filters to narrow down the search space.
            top_k: The maximum number of documents to retrieve.
            scale_score: Whether to scale the scores of the retrieved documents or not.
            return_embedding: Whether to return the embedding of the retrieved Documents.

        Raises:
            ValueError: If `document_store` is not an instance of `Neo4jDocumentStore`.
        """

        if not isinstance(document_store, Neo4jDocumentStore):
            msg = "document_store must be an instance of Neo4jDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store

        self._filters = filters
        self._top_k = top_k
        self._scale_score = scale_score
        self._return_embedding = return_embedding

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        data = default_to_dict(
            self,
            document_store=self._document_store,
            filters=self._filters,
            top_k=self._top_k,
            scale_score=self._scale_score,
            return_embedding=self._return_embedding,
        )
        data["init_parameters"]["document_store"] = self._document_store.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neo4jEmbeddingRetriever":
        """
        Deserialize this component from a dictionary.
        """
        document_store = Neo4jDocumentStore.from_dict(data["init_parameters"]["document_store"])
        data["init_parameters"]["document_store"] = document_store
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        return_embedding: Optional[bool] = None,
    ):
        """
        Run the Embedding Retriever on the given input data.

        Args:
            query_embedding: Embedding of the query.
            filters: A dictionary with filters to narrow down the search space.
            top_k: The maximum number of documents to return.
            scale_score: Whether to scale the scores of the retrieved documents or not.
            return_embedding: Whether to return the embedding of the retrieved Documents.

        Returns:
            The retrieved documents.
        """
        docs = self._document_store.query_by_embedding(
            query_embedding=query_embedding,
            filters=filters or self._filters,
            top_k=top_k or self._top_k,
            scale_score=scale_score or self._scale_score,
            return_embedding=return_embedding or self._return_embedding,
        )

        return {"documents": docs}


@component
class Neo4jDynamicDocumentRetriever:
    """
    A component for retrieving Documents from Neo4j database using plain Cypher query.

    This component gives flexible way to retrieve data from Neo4j by running arbitrary Cypher query along with query
    parameters. Query parameters can be supplied in a pipeline from other components (or pipeline data).

    See the following documentation on how to compose Cypher queries with parameters:

    - [Overview of Cypher query syntax](https://neo4j.com/docs/cypher-manual/current/queries/)
    - [Cypher Query Parameters](https://neo4j.com/docs/cypher-manual/current/syntax/parameters/)

    Above are resources which will help understand better Cypher query syntax and parameterization. Under the hood
    [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/) is used to query database and fetch results.
    You might be interested in the following documentation:

    - [Query the database](https://neo4j.com/docs/python-manual/current/query-simple/)
    - [Query parameters](https://neo4j.com/docs/python-manual/current/query-simple/#query-parameters)
    - [Data types and mapping to Cypher types](https://neo4j.com/docs/python-manual/current/data-types/)

    Note:
        Please consider data types mappings in Cypher query when working with parameters. Neo4j Python Driver handles
        type conversions/mappings. Specifically you can figure out in the documentation of the driver how to work with
        temporal types (e.g. `DateTime`).

    Query execution results will be mapped/converted to `haystack.Document` type. See more details in the
    [RETURN clause](https://neo4j.com/docs/cypher-manual/current/clauses/return/) documentation. There are two
    ways how Documents are being composed from query results.

    (1) Converting documents from [nodes](https://neo4j.com/docs/cypher-manual/current/clauses/return/#return-nodes)

    ```py title="Convert Neo4j `node` to `haystack.Document`"
    client_config = Neo4jClientConfig(
        "bolt://localhost:7687", database="neo4j", username="neo4j", password="passw0rd"
    )

    retriever = Neo4jDynamicDocumentRetriever(
        client_config=client_config, doc_node_name="doc", verify_connectivity=True
    )

    result = retriever.run(
        query="MATCH (doc:Document) WHERE doc.year > $year OR doc.year is NULL RETURN doc",
        parameters={"year": 2020}
    )
    documents: List[Document] = result["documents"]
    ```

    Please notice how `doc_node_name` attribute assumes `"doc"` node is going to be returned from the query.
    `Neo4jDynamicDocumentRetriever` will convert properties of the node (e.g. `id`, `content` etc) to
    `haystack.Document` type.

    (2) Converting documents from query output keys (e.g. [column aliases](https://neo4j.com/docs/cypher-manual/current/clauses/return/#return-column-alias))

    You might want to run a complex query which aggregates information from multiple sources (nodes) in Neo4j. In such
    case you can compose final Document from several dta points.

    ```py title="Convert Neo4j `node` to `haystack.Document`"
    # Configuration with default settings
    client_config=Neo4jClientConfig()

    retriever = Neo4jDynamicDocumentRetriever(client_config=client_config, compose_doc_from_result=True)

    result = retriever.run(
        query=(
            "MATCH (doc:Document) "
            "WHERE doc.year > $year OR doc.year is NULL "
            "RETURN doc.id as id, doc.content as content, doc.year as year"
        ),
        parameters={"year": 2020},
    )
    documents: List[Document] = result["documents"]
    ```

    The above will produce Documents with `id`, `content` and `year`(meta) fields. Please notice
    `compose_doc_from_result` is set to `True` to enable such Document construction behavior.

    Below is an example of a pipeline which explores all ways how parameters could be supplied to the
    `Neo4jDynamicDocumentRetriever` component in the pipeline.

    ```py
    @component
    class YearProvider:
        @component.output_types(year_start=int, year_end=int)
        def run(self, year_start: int, year_end: int):
            return {"year_start": year_start, "year_end": year_end}

    # Configuration with default settings
    client_config=Neo4jClientConfig()

    retriever = Neo4jDynamicDocumentRetriever(
        client_config=client_config,
        runtime_parameters=["year_start", "year_end"],
    )

    query = (
        "MATCH (doc:Document) "
        "WHERE (doc.year >= $year_start and doc.year <= $year_end) AND doc.month = $month"
        "RETURN doc LIMIT $num_return"
    )

    pipeline = Pipeline()
    pipeline.add_component("year_provider", YearProvider())
    pipeline.add_component("retriever", retriever)
    pipeline.connect("year_provider.year_start", "retriever.year_start")
    pipeline.connect("year_provider.year_end", "retriever.year_end")

    result = pipeline.run(
        data={
            "year_provider": {"year_start": 2020, "year_end": 2021},
            "retriever": {
                "query": query,
                "parameters": {
                    "month": "02",
                    "num_return": 2,
                },
            },
        }
    )

    documents = result["retriever"]["documents"]
    ```

    Please notice the following from the example above:

    - `runtime_parameters` is a list of parameter names which are going to be input slots when connecting components
        in a pipeline. In our case `year_start` and `year_end` parameters flow from the `year_provider` component into
        `retriever`. The `query` uses those parameters in the `WHERE` clause.
    - `pipeline.run` specifies additional parameters to the `retriever` component which can be referenced in the
        `query`. If parameter names clash those provided in the pipeline's data take precedence.
    """

    def __init__(
        self,
        client_config: Neo4jClientConfig,
        query: Optional[str] = None,
        runtime_parameters: Optional[List[str]] = None,
        doc_node_name: Optional[str] = "doc",
        compose_doc_from_result: Optional[bool] = False,
        verify_connectivity: Optional[bool] = False,
    ):
        """
        Create a Neo4jDynamicDocumentRetriever component.

        Args:
            client_config: Neo4j client configuration to connect to database (e.g. credentials and connection settings).
            query: Optional Cypher query for document retrieval. If `None` should be provided as component input.
            runtime_parameters: list of input parameters/slots for connecting components in a pipeline.
            doc_node_name: the name of the variable which is returned from Cypher query which contains Document
                attributes (e.g. `id`, `content`, `meta` fields).
            compose_doc_from_result: If `True` Document attributes will be constructed from Cypher query outputs (keys).
                `doc_node_name` setting will be ignored in this case.
            verify_connectivity: If `True` will verify connectivity with Neo4j database configured by `client_config`.

        Raises:
            ComponentError: In case neither `compose_doc_from_result` nor `doc_node_name` are defined.
        """
        if not compose_doc_from_result and not doc_node_name:
            raise ComponentError(
                "Please specify how Document is being composed out of Neo4j query response. "
                "With `compose_doc_from_result` set to `True` documents will be created out of properties/keys "
                "returned by the query."
            )

        self._client_config = client_config
        self._query = query
        self._runtime_parameters = runtime_parameters or []
        self._doc_node_name = doc_node_name
        self._compose_doc_from_result = compose_doc_from_result
        self._verify_connectivity = verify_connectivity

        self._neo4j_client = Neo4jClient(client_config)

        # setup inputs
        kwargs_input_slots = dict.fromkeys(self._runtime_parameters, Optional[Any])
        component.set_input_types(self, **kwargs_input_slots)

        # setup outputs
        component.set_output_types(self, documents=List[Document])

        if verify_connectivity:
            self._neo4j_client.verify_connectivity()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        data = default_to_dict(
            self,
            query=self._query,
            runtime_parameters=self._runtime_parameters,
            doc_node_name=self._doc_node_name,
            compose_doc_from_result=self._compose_doc_from_result,
            verify_connectivity=self._verify_connectivity,
        )

        data["init_parameters"]["client_config"] = self._client_config.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neo4jDynamicDocumentRetriever":
        """
        Deserialize this component from a dictionary.
        """
        client_config = Neo4jClientConfig.from_dict(data["init_parameters"]["client_config"])
        data["init_parameters"]["client_config"] = client_config
        return default_from_dict(cls, data)

    def run(self, query: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None, **kwargs: Dict[str, Any]):
        """
        Runs the arbitrary Cypher `query` with `parameters` and returns Documents.

        Args:
            query: Cypher query to run.
            parameters: Cypher query parameters which can be used as placeholders in the `query`.
            kwargs: Arbitrary parameters supplied in a pipeline execution from other component's output slots, e.g.
                `pipeline.connect("year_provider.year_start", "retriever.year_start")`, where `year_start` will be part
                of `kwargs`.

        Returns:
            Retrieved documents.
        """
        query = query or self._query
        if query is None:
            raise ValueError(
                "`query` is mandatory input and should be provided either in component's constructor, pipeline input or"
                "connection"
            )
        kwargs = kwargs or {}
        parameters = parameters or {}
        parameters_combined = {**kwargs, **parameters}

        documents: List[Document] = []
        neo4j_query_result = self._neo4j_client.query_nodes(query, parameters_combined)

        for record in neo4j_query_result:
            data = record.data()
            document_dict = data if self._compose_doc_from_result else data.get(cast(str, self._doc_node_name))
            documents.append(Document.from_dict(document_dict))

        return {"documents": documents}
