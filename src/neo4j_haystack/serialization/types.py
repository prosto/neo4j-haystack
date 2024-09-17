from typing import Any, Protocol


class QueryParametersMarshaller(Protocol):
    """
    A Protocol to be used by marshaller implementations which convert Neo4j query parameters to appropriate types.
    """

    def supports(self, obj: Any) -> bool:
        pass

    def marshal(self, obj: Any) -> Any:
        pass
