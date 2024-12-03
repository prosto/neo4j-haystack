import logging
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from typing_extensions import NotRequired, TypedDict

from neo4j_haystack.client import Neo4jClient, Neo4jClientConfig
from neo4j_haystack.serialization import (
    Neo4jQueryParametersMarshaller,
    QueryParametersMarshaller,
)

logger = logging.getLogger(__name__)


class QueryResult(TypedDict):
    """
    Query execution outputs for the `Neo4jQueryReader` component.
    """

    records: NotRequired[List[Dict[str, Any]]]
    first_record: NotRequired[Optional[Dict[str, Any]]]
    error: NotRequired[Exception]
    error_message: NotRequired[str]


@component
class Neo4jQueryReader:
    """
    A component for reading arbitrary data from Neo4j database using plain Cypher query.

    This component gives flexible way to read data from Neo4j by running custom Cypher query along with query
    parameters. Query parameters can be supplied in a pipeline from other components (or pipeline inputs).
    You could use such queries to read data from Neo4j to enhance your RAG pipelines. For example a
    prompt to LLM can produce Cypher query based on given context and then `Neo4jQueryReader` can be used to run the
    query and extract results. [OutputAdapter](https://docs.haystack.deepset.ai/docs/outputadapter) component might
    become handy in such scenarios - it can be used as a connection from the `Neo4jQueryReader` to convert (transform)
    results accordingly.

    Note:
        Please consider [data types mappings](https://neo4j.com/docs/api/python-driver/current/api.html#data-types) in \
        Cypher query when working with query parameters. Neo4j Python Driver handles type conversions/mappings.
        Specifically you can figure out in the documentation of the driver how to work with temporal types.

    ```py title="Example: Find a Document node with Neo4jQueryReader and extract data"
    from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
    from neo4j_haystack.components.neo4j_query_reader import Neo4jQueryReader

    client_config = Neo4jClientConfig("bolt://localhost:7687", database="neo4j", username="neo4j", password="passw0rd")

    reader = Neo4jQueryReader(client_config=client_config, runtime_parameters=["year"])

    # Get all documents with "year"=2020 and return "name" and "embedding" attributes for each found record
    result = reader.run(
        query=("MATCH (doc:`Document`) WHERE doc.year=$year RETURN doc.name as name, doc.embedding as embedding"),
        year=2020,
    )
    ```

    Output:
        `>>> {'records': [{'name': 'name_0', 'embedding': [...]}, {'name': 'name_1', 'embedding': [...]}, \
        {'name': 'name_2', 'embedding': [...]}], 'first_record': {'name': 'name_0', 'embedding': [...]}}`

    The above result contains the following output:

    - `records` - A list of dictionaries, will have all the records returned by Cypher query. You can control record
        outputs as per your needs. For example an aggregation function could be used to return a single result.
        In such case there will be one record in the `records` list.
    - `first_record` - In case the `records` contains just  one item, `first_record` will have the first record
        from the list (put simply, first_record=records[0]). It was introduced as a syntax convenience.

    If your Cypher query produces an error (e.g. invalid syntax) you could use that in `Loop-Based Auto-Correction`
    pipelines to ask LLM to auto correct the query based on the error message, afterwards run the query again.

    ```py title="Example: Output error with Neo4jQueryReader"
    from neo4j_haystack.client.neo4j_client import Neo4jClientConfig
    from neo4j_haystack.components.neo4j_query_reader import Neo4jQueryReader

    client_config = Neo4jClientConfig("bolt://localhost:7687", database="neo4j", username="neo4j", password="passw0rd")

    reader = Neo4jQueryReader(client_config=client_config, raise_on_failure=False)

    # Intentionally introduce error in Cypher query (see "RETURN_")
    result = reader.run(
        query=("MATCH (doc:`Document` {name: $name}) RETURN_ doc.name as name, doc.year as year"),
        parameters={"name": "name_1"},
    )
    ```

    Output:
        `>>> {'error_message': 'Invalid input \'RETURN_\'...', 'error': <Exception>}`

    The `error_message` output can be used in your pipeline to deal with Cypher query error (e.g. auto correction)

    When configuring [Query parameters](https://neo4j.com/docs/python-manual/current/query-simple/#query-parameters) \
    for `Neo4jQueryReader` component, consider the following:

    - Parameters can be provided at the component creation time, see `parameters`
    - In RAG pipeline runtime parameters could be connected from other components.
      Make sure during creation time to specify which `runtime_parameters` are expected.

    Important:
        At the moment parameters support simple data types, dictionaries and python dataclasses (which can be converted
        to `dict`). For example `haystack.ChatMessage` instance is a valid query parameter input. If you supply custom
        classes as query parameters, e.g. \
        `Neo4jQueryReader(client_config=client_config).run(parameters={"obj": <instance of custom class>})` it will
        result in error. In such rare cases `query_parameters_marshaller` attribute can be used to provide a
        custom marshaller implementation for the type being used as query parameter value.
    """

    def __init__(
        self,
        client_config: Neo4jClientConfig,
        query: Optional[str] = None,
        runtime_parameters: Optional[List[str]] = None,
        verify_connectivity: Optional[bool] = False,
        raise_on_failure: bool = False,
        query_parameters_marshaller: Optional[QueryParametersMarshaller] = None,
    ):
        """
        Creates a Neo4jDocumentReader component.

        Args:
            client_config: Neo4j client configuration to connect to database (e.g. credentials and connection settings).
            query: Optional Cypher query if known at component creation time. If `None` should be provided as component
                input.
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
        component.set_output_types(
            self, records=List[Dict[str, Any]], expanded_record=Optional[Dict[str, Any]], error=Optional[str]
        )

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
    def from_dict(cls, data: Dict[str, Any]) -> "Neo4jQueryReader":
        """
        Deserialize this component from a dictionary.
        """
        client_config = Neo4jClientConfig.from_dict(data["init_parameters"]["client_config"])
        data["init_parameters"]["client_config"] = client_config
        return default_from_dict(cls, data)

    def run(self, query: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None, **kwargs) -> QueryResult:
        """
        Runs the arbitrary Cypher `query` with `parameters` to read data from Neo4j.

        Args:
            query: Cypher query to run.
            parameters: Cypher query parameters which can be used as placeholders in the `query`.
            kwargs: Arbitrary parameters supplied in a pipeline execution from other component's output slots, e.g.
                `pipeline.connect("year_provider.year_start", "reader.year_start")`, where `year_start` will be part
                of `kwargs`.

        Returns:
            Output: Records returned from Cypher query in case request was successful or error message if there was an
                error during Cypher query execution (`raise_on_failure` should be `False`).

                ```py title="Example: Output with records"
                {'records': [{...}, {...}], 'first_record': {...}}
                ```

                where:

                - `records` - List of records returned (e.g. using `RETURN` statement) by Cypher query
                - `first_record` - First record from the `records` list if any

                ```py title="Example: Output with error"
                {'error_message': 'Invalid Cypher syntax...', 'error': <Exception>}
                ```

                where:

                - `error_message` - Error message returned by Neo4j in case Cypher query is invalid
                - `error` - Original Exception which was triggered by Neo4j (containing the `error_message`)
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
            _, records = self._neo4j_client.execute_read(
                query,
                parameters=self._serialize_parameters(parameters_combined),
            )

            return {"records": records, "first_record": records[0] if len(records) > 0 else None}
        except Exception as ex:
            if self._raise_on_failure:
                logger.error("Couldn't execute Neo4j read query %s", ex)
                raise ex

            return {
                "error": ex,
                "error_message": str(ex),
            }

    def _serialize_parameters(self, parameters: Any) -> Any:
        """
        Serializes `parameters` into a data structure which can be accepted by Neo4j Python Driver (and a Cypher query
        respectively). See \
            [Neo4jQueryParametersMarshaller][neo4j_haystack.serialization.query_parameters_marshaller.Neo4jQueryParametersMarshaller]
            for more details.
        """
        return self._query_parameters_marshaller.marshal(parameters)
