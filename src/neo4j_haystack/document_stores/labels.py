from typing import Dict, List, Optional, Union

from haystack.schema import FilterType, Label


class Neo4jDocumentStoreLabels:
    """
    A mock implementation of "label" related methods from [BaseDocumentStore API](https://docs.haystack.deepset.ai/reference/base-document-store-api#basedocumentstore)
    """

    def get_all_labels(
        self,
        index: Optional[str] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> List[Label]:
        raise NotImplementedError("Neo4jDocumentStore does not support labels")

    def get_label_count(self, index: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> int:
        raise NotImplementedError("Neo4jDocumentStore does not support labels")

    def write_labels(
        self,
        labels: Union[List[Label], List[dict]],
        index: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("Neo4jDocumentStore does not support labels")

    def delete_labels(
        self,
        index: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filters: Optional[FilterType] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        raise NotImplementedError("Neo4jDocumentStore does not support labels")
