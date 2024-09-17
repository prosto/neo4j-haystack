from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from haystack import Document

from neo4j_haystack.serialization.types import QueryParametersMarshaller


class DocumentQueryParametersMarshaller(QueryParametersMarshaller):
    """
    A marshaller which converts `haystack.Document` to a dictionary when it is used as query parameter in Cypher query

    Haystack's native `to_dict` method is called to produce a flattened dictionary of Document data along with its meta
    fields.
    """

    def supports(self, obj: Any) -> bool:
        """
        Checks if given object is `haystack.Document` instance
        """
        return isinstance(obj, Document)

    def marshal(self, obj: Any) -> Dict[str, Any]:
        """
        Converts `haystack.Document` to dictionary so it could be used as Cypher query parameter
        """
        return obj.to_dict(flatten=True)


class DataclassQueryParametersMarshaller(QueryParametersMarshaller):
    """
    A marshaller which converts a `dataclass` to a dictionary when encountered as Cypher query parameter.
    """

    def supports(self, obj: Any) -> bool:
        """
        Checks if given object is a python `dataclass` instance
        """
        return is_dataclass(obj) and not isinstance(obj, type)

    def marshal(self, obj: Any) -> Dict[str, Any]:
        """
        Converts `dataclass` to dictionary so it could be used as Cypher query parameter.
        """
        return asdict(obj)


class Neo4jQueryParametersMarshaller(QueryParametersMarshaller):
    """
    The marshaller converts Cypher query parameters to types which can be consumed in Neo4j query execution. In some
    cases query parameters contain complex data types (e.g. dataclasses) which can be converted to `dict` types and
    thus become eligible for running a Cypher query:

    ```py title="Example: Running a query with a dataclass parameter"
    from dataclasses import dataclass
    from neo4j_haystack.client import Neo4jClient, Neo4jClientConfig
    from neo4j_haystack.serialization import Neo4jQueryParametersMarshaller

    @dataclass
    class YearInfo:
        year: int = 2024

    neo4j_config = Neo4jClientConfig("bolt://localhost:7687", database="neo4j", username="neo4j", password="passw0rd")
    neo4j_client = Neo4jClient(client_config=neo4j_config)

    marshaller = Neo4jQueryParametersMarshaller()
    query_params = marshaller.marshall({"year_info": YearInfo()})

    neo4j_client.execute_read(
        "MATCH (doc:`Document`) WHERE doc.year=$year_info.year RETURN doc",
        parameters=query_params
    )
    ```
    The above example would fail without marshaller. With marshaller `query_params` will become
    `:::py {"year_info": {"year": 2024}}` which is eligible input for Neo4j Driver.
    """

    def __init__(self):
        self._marshallers = [DocumentQueryParametersMarshaller(), DataclassQueryParametersMarshaller()]

    def supports(self, _obj: Any) -> bool:
        """
        Supports conversion of any type
        """
        return True

    def marshal(self, data) -> Any:
        if isinstance(data, dict):
            return {key: self.marshal(value) for key, value in data.items()}
        elif isinstance(data, (list, tuple, set)):
            return [self.marshal(value) for value in data]

        # Find a marshaller which supports given type of parameter otherwise return parameter as is
        marshaller = next((m for m in self._marshallers if m.supports(data)), None)

        return self.marshal(marshaller.marshal(data)) if marshaller else data
