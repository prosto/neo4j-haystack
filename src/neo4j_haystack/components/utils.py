from collections.abc import MutableMapping
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Protocol

from haystack import Document


def _flatten_dict_gen(d, parent_key: str, sep: str):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from _flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def _flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    return dict(_flatten_dict_gen(d, parent_key, sep))


class Neo4jQueryParameterMarshaller(Protocol):
    def supports(self, obj: Any) -> bool:
        pass

    def marshal(self, obj: Any) -> Dict[str, Any]:
        pass


class DocumentParameterMarshaller:
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


class DataclassParameterMarshaller:
    def supports(self, obj: Any) -> bool:
        """
        Checks if given object is a python `dataclass` instance
        """
        return is_dataclass(obj) and not isinstance(obj, type)

    def marshal(self, obj: Any) -> Dict[str, Any]:
        """
        Converts `dataclass` to dictionary so it could be used as Cypher query parameter.
        Nested attributes will be flattened and separated by dot (".")
        """
        return _flatten_dict(asdict(obj))
