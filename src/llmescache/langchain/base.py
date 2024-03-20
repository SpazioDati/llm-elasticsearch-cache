from abc import abstractmethod
from typing import Dict, Any

import elasticsearch
from elasticsearch import Elasticsearch


class ElasticsearchIndexer:
    """Mixin for Elasticsearch clients"""

    _es_client: Elasticsearch
    _es_index: str

    def _manage_index(self):
        """Write or update an index according to the default mapping"""
        if not self._es_client.ping():
            raise elasticsearch.exceptions.ConnectionError(
                "Elasticsearch cluster is not available, not able to set up the cache store."
            )
        self._is_alias = False
        if self._es_client.indices.exists_alias(name=self._es_index):
            self._is_alias = True
        elif not self._es_client.indices.exists(index=self._es_index):
            self._es_client.indices.create(index=self._es_index, body=self.mapping)
            return
        self._es_client.indices.put_mapping(
            index=self._es_index, body=self.mapping["mappings"]
        )

    @property
    @abstractmethod
    def mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {}
