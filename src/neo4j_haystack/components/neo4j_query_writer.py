import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict

from haystack import component, default_from_dict, default_to_dict
from neo4j import ResultSummary

from neo4j_haystack.client import Neo4jClient, Neo4jClientConfig
from neo4j_haystack.components.utils import (
    DataclassParameterMarshaller,
    DocumentParameterMarshaller,
    Neo4jQueryParameterMarshaller,
)

logger = logging.getLogger(__name__)


class QueryResult(TypedDict):
    query_type: Optional[str]
    query_status: Literal["success", "error"]


@component
class Neo4jQueryWriter:
    """
    A component for writing arbitrary data to Neo4j database using plain Cypher query.

    This component gives flexible way to write data to Neo4j by running arbitrary Cypher query along with query
    parameters. Query parameters can be supplied in a pipeline from other components (or pipeline data).
    You could use such queries to write Documents with additional graph nodes for a more complex RAG scenarios.

    See the following documentation on how to compose Cypher queries with parameters:

    - [Overview of Cypher query syntax](https://neo4j.com/docs/cypher-manual/current/queries/)
    - [Cypher Query Parameters](https://neo4j.com/docs/cypher-manual/current/syntax/parameters/)

    Above are resources which will help understand better Cypher query syntax and parameterization. Under the hood
    [Neo4j Python Driver](https://neo4j.com/docs/python-manual/current/) is used to write data to Neo4j.
    Check the following documentation:

    - [Query the database](https://neo4j.com/docs/python-manual/current/query-simple/)
    - [Query parameters](https://neo4j.com/docs/python-manual/current/query-simple/#query-parameters)
    - [Data types and mapping to Cypher types](https://neo4j.com/docs/python-manual/current/data-types/)

    Note:
        Please consider data types mappings in Cypher query when working with parameters. Neo4j Python Driver handles
        type conversions/mappings. Specifically you can figure out in the documentation of the driver how to work with
        temporal types (e.g. `DateTime`).

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
        `>>> {'result_available_after': 4, 'result_consumed_after': 0, 'query_type': 'w'}`

    The above example shows the flexibility of the `Neo4jQueryWriter` component:

    - Cypher query can practically write any data to Neo4j depending on your use case
    - Parameters can be provided at the component creation time, see `parameters`
    - In RAG pipeline runtime parameters could be connected from other components.
      Make sure during creation time to specify which `runtime_parameters` are expected.

    Important:
        At the moment parameters support simple data types, dictionaries (see `doc_meta` in the example above) and
        python dataclasses (which can be converted to `dict`). For example `haystack.Document` or `haystack.ChatMessage`
        instances are valid query parameter inputs. However, currently Neo4j Python Driver does not support
        dataclasses/types with nested attributes. `haystack.Document` when serialized to dictionary flattens its meta
        attributes. `haystack.ChatMessage`, for example, has custom serializer `DataclassParameterMarshaller` which
        automatically flattens nested attributes like `meta`. So when you pass a \
        `ChatMessage.from("Hello", meta={"year": 2017})` instance to the query as a parameter its dictionary form will
        be as follows: `{role: 'user', 'meta.year': 2017}`
    """

    def __init__(
        self,
        client_config: Neo4jClientConfig,
        runtime_parameters: Optional[List[str]] = None,
        verify_connectivity: Optional[bool] = False,
        raise_on_failure: bool = True,
    ):
        """
        Create a Neo4jDocumentWriter component.

        Args:
            client_config: Neo4j client configuration to connect to database (e.g. credentials and connection settings).
            runtime_parameters: list of input parameters/slots for connecting components in a pipeline.
            verify_connectivity: If `True` will verify connectivity with Neo4j database configured by `client_config`.
            raise_on_failure: If `True` raises an exception if it fails to execute given Cypher query.
        """
        self._client_config = client_config
        self._runtime_parameters = runtime_parameters or []
        self._verify_connectivity = verify_connectivity
        self._raise_on_failure = raise_on_failure

        self._neo4j_client = Neo4jClient(client_config)
        self._marshallers: List[Neo4jQueryParameterMarshaller] = [
            DocumentParameterMarshaller(),
            DataclassParameterMarshaller(),
        ]

        # setup inputs
        run_input_slots = {"query": str, "parameters": Optional[Dict[str, Any]]}
        kwargs_input_slots = {param: Optional[Any] for param in self._runtime_parameters}
        component.set_input_types(self, **run_input_slots, **kwargs_input_slots)

        # setup outputs
        component.set_output_types(self, query_type=Optional[str], query_status=str)

        if verify_connectivity:
            self._neo4j_client.verify_connectivity()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        data = default_to_dict(
            self,
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

    def run(self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs) -> QueryResult:
        """
        Runs the arbitrary Cypher `query` with `parameters` to write data to Neo4j.

        Once data is written to Neo4j the component returns back some execution stats.

        Args:
            query: Cypher query to run.
            parameters: Cypher query parameters which can be used as placeholders in the `query`.
            kwargs: Arbitrary parameters supplied in a pipeline execution from other component's output slots, e.g.
                `pipeline.connect("year_provider.year_start", "writer.year_start")`, where `year_start` will be part
                of `kwargs`.

        Returns:
            Output: Query execution stats.

                Example: `:::py {'result_available_after': 4, 'result_consumed_after': 0, 'query_type': 'w'}`

                where:

                - `result_available_after` - The time it took for the server to have the result available. (in 'ms')
                - `result_consumed_after` - The time it took for the server to consume the result. (in 'ms')
                - `query_type` - A string that describes the type of query (e.g. 'r', 'rw', 'w', 's')
        """
        kwargs = kwargs or {}
        parameters = parameters or {}
        parameters_combined = {**kwargs, **parameters}

        try:
            result_summary, _ = self._neo4j_client.execute_write(
                query, parameters=self._serialize_parameters(parameters_combined)
            )

            return self._query_result(result_summary)
        except Exception as ex:
            if self._raise_on_failure:
                raise ex
            logger.error("Couldn't execute Neo4j write query %s", ex)
            return self._query_result()

    def _serialize_parameters(self, parameters: Any) -> Any:
        """
        Serializes `parameters` into data structure which can be accepted by Neo4j Python Driver (and a Cypher query
        respectively). See the following marshallers with implementation details:

        - [DocumentParameterMarshaller][neo4j_haystack.components.utils.DocumentParameterMarshaller] - converts
            `haystack.Document` instances to a dictionary
        - [DataclassParameterMarshaller][neo4j_haystack.components.utils.DataclassParameterMarshaller] - converts
            dataclasses to a dictionary with flattened nested attributes

        Args:
            parameters: Any object which is passed as a Cypher query parameter

        Returns:
            Serialized parameters
        """
        if isinstance(parameters, dict):
            return {key: self._serialize_parameters(value) for key, value in parameters.items()}
        elif isinstance(parameters, (list, tuple, set)):
            return [self._serialize_parameters(value) for value in parameters]

        # Find a marshaller which supports given type of parameter otherwise return parameter as is
        marshaller = next((m for m in self._marshallers if m.supports(parameters)), None)
        return marshaller.marshal(parameters) if marshaller else parameters

    def _query_result(self, result_summary: Optional[ResultSummary] = None) -> QueryResult:
        if result_summary:
            return {
                "query_type": result_summary.query_type,
                "query_status": "success",
            }
        else:
            return {
                "query_type": "w",
                "query_status": "error",
            }
