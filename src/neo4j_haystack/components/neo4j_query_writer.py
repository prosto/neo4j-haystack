import logging
from typing import Any, Dict, List, Literal, Optional

from haystack import component, default_from_dict, default_to_dict
from typing_extensions import NotRequired, TypedDict

from neo4j_haystack.client import Neo4jClient, Neo4jClientConfig
from neo4j_haystack.serialization import (
    Neo4jQueryParametersMarshaller,
    QueryParametersMarshaller,
)

logger = logging.getLogger(__name__)


class QueryResult(TypedDict):
    query_status: Literal["success", "error"]
    error_message: NotRequired[str]
    error: NotRequired[Exception]


@component
class Neo4jQueryWriter:
    """
    A component for writing arbitrary data to Neo4j database using plain Cypher query.

    This component gives flexible way to write data to Neo4j by running arbitrary Cypher query with
    parameters. Query parameters can be supplied in a pipeline from other components (or pipeline data).
    You could use such queries to write Documents with additional graph nodes for a more complex RAG scenarios.
    The difference between [DocumentWriter](https://docs.haystack.deepset.ai/docs/documentwriter) and `Neo4jQueryWriter`
    is that the latter can write any data to Neo4j - not just Documents.

    Note:
        Please consider [data types mappings](https://neo4j.com/docs/api/python-driver/current/api.html#data-types) in \
        Cypher query when working with query parameters. Neo4j Python Driver handles type conversions/mappings.
        Specifically you can figure out in the documentation of the driver how to work with temporal types.

    ```py title="Example: Creating a Document node with Neo4jQueryWriter"
    from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
    from neo4j_haystack.components.neo4j_query_writer import Neo4jQueryWriter

    client_config = Neo4jClientConfig("bolt://localhost:7687", database="neo4j", username="neo4j", password="passw0rd")

    doc_meta = {"year": 2020, "source_url": "https://www.deepset.ai/blog"}

    writer = Neo4jQueryWriter(client_config=client_config, verify_connectivity=True, runtime_parameters=["doc_meta"])

    result = writer.run(
        query=(
            "MERGE (doc:`Document` {id: $doc_id})"
            "SET doc += {id: $doc_id, content: $content, year: $doc_meta.year, source_url: $doc_meta.source_url}"
        ),
        parameters={"doc_id": "123", "content": "beautiful graph"},
        doc_meta=doc_meta
    )
    ```

    Output:
        `>>> {'query_status': 'success'}`

    In case query execution results in error and `raise_on_failure=False` the output will contain the error, e.g.:

    Output:
        `>>> {'query_status': 'error', 'error_message': 'Invalid cypher syntax', error: <Exception>}`

    In RAG pipeline runtime parameters could be connected from other components. Make sure during component creation to
    specify which `runtime_parameters` are expected to become as input slots for the component. In the example above
    `doc_meta` can be connected , e.g. `pipeline.connect("other_component.output", "writer.doc_meta")`.

    Important:
        At the moment parameters support simple data types, dictionaries (see `doc_meta` in the example above) and
        python dataclasses (which can be converted to `dict`). For example `haystack.Document` or `haystack.ChatMessage`
        instances are valid query parameter inputs. However, currently Neo4j Python Driver does not convert dataclasses
        to dictionaries automatically for us. By default \
            [Neo4jQueryParametersMarshaller][neo4j_haystack.serialization.query_parameters_marshaller.Neo4jQueryParametersMarshaller]
        is used to handle such conversions. You can change this logic by creating your own marshaller (see the
        `query_parameters_marshaller` attribute)
    """

    def __init__(
        self,
        client_config: Neo4jClientConfig,
        query: Optional[str] = None,
        runtime_parameters: Optional[List[str]] = None,
        verify_connectivity: Optional[bool] = False,
        raise_on_failure: bool = True,
        query_parameters_marshaller: Optional[QueryParametersMarshaller] = None,
    ):
        """
        Create a Neo4jDocumentWriter component.

        Args:
            client_config: Neo4j client configuration to connect to database (e.g. credentials and connection settings).
            query: Optional Cypher query for document retrieval. If `None` should be provided as component input.
            runtime_parameters: list of input parameters/slots for connecting components in a pipeline.
            verify_connectivity: If `True` will verify connectivity with Neo4j database configured by `client_config`.
            raise_on_failure: If `True` raises an exception if it fails to execute given Cypher query.
            query_parameters_marshaller: Marshaller responsible for converting query parameters which can be used in
                Cypher query, e.g. python dataclasses to be converted to dictionary. `Neo4jQueryParametersMarshaller`
                is the default marshaller implementation.
        """
        self._client_config = client_config
        self._query = query
        self._runtime_parameters = runtime_parameters or []
        self._verify_connectivity = verify_connectivity
        self._raise_on_failure = raise_on_failure

        self._neo4j_client = Neo4jClient(client_config)
        self._query_parameters_marshaller = query_parameters_marshaller or Neo4jQueryParametersMarshaller()

        # setup inputs
        kwargs_input_slots = {param: Optional[Any] for param in self._runtime_parameters}
        component.set_input_types(self, **kwargs_input_slots)

        # setup outputs
        component.set_output_types(self, query_status=str, error=Optional[Exception], error_message=Optional[str])

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
            verify_connectivity=self._verify_connectivity,
            raise_on_failure=self._raise_on_failure,
        )

        data["init_parameters"]["client_config"] = self._client_config.to_dict()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neo4jQueryWriter":
        """
        Deserialize this component from a dictionary.
        """
        client_config = Neo4jClientConfig.from_dict(data["init_parameters"]["client_config"])
        data["init_parameters"]["client_config"] = client_config
        return default_from_dict(cls, data)

    def run(self, query: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None, **kwargs) -> QueryResult:
        """
        Runs the arbitrary Cypher `query` with `parameters` to write data to Neo4j.

        Once data is written to Neo4j the component returns back some execution stats.

        Args:
            query: Cypher query to run.
            parameters: Cypher query parameters which can be used as placeholders in the `query`.
            kwargs: Runtime parameters from connected components in a pipeline, e.g.
                `pipeline.connect("year_provider.year_start", "writer.year_start")`, where `year_start` will be part
                of `kwargs`.

        Returns:
            Output: Query execution stats.

                Example: `:::py {'query_status': 'success'}`
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

        try:
            self._neo4j_client.execute_write(
                query,
                parameters=self._serialize_parameters(parameters_combined),
            )

            return {"query_status": "success"}
        except Exception as ex:
            if self._raise_on_failure:
                logger.error("Couldn't execute Neo4j write query %s", ex)
                raise ex

            return {"query_status": "error", "error_message": str(ex), "error": ex}

    def _serialize_parameters(self, parameters: Any) -> Any:
        """
        Serializes `parameters` into a data structure which can be accepted by Neo4j Python Driver (and a Cypher query
        respectively). See \
            [Neo4jQueryParametersMarshaller][neo4j_haystack.serialization.query_parameters_marshaller.Neo4jQueryParametersMarshaller]
            for more details.
        """
        return self._query_parameters_marshaller.marshal(parameters)
