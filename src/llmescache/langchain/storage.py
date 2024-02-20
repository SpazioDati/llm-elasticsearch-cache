import hashlib
from typing import List, Optional, Iterator, Sequence, Tuple, Any, Dict

import elasticsearch
from langchain_core.stores import BaseStore


class ElasticsearchStore(BaseStore[str, List[float]]):
    def __init__(
        self,
        es_client: elasticsearch.Elasticsearch,
        es_index: str,
        store_input: bool = True,
        store_timestamp: bool = True,
        namespace: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Elasticsearch cache store by specifying the index/alias
        to use and determining which additional information (like input, timestamp, input parameters,
        and any other metadata) should be stored in the cache.

        Args:
            es_client (Elasticsearch): The Elasticsearch client to use for the cache store.
            es_index (str): The name of the index or the alias to use for the cache store.
            If they do not exist an index is created, according to the default mapping defined by `mapping` property.
            store_input (bool): Whether to store the LLM input in the cache, i.e., the input prompt. Default to True.
            store_timestamp (bool): Whether to store the datetime in the cache, i.e., the time of the
                first request for a LLM input. Default to True.
            namespace (Optional[str]): If provided, all keys will be prefixed with this namespace, Default to None.
            metadata (Optional[dict]): Additional metadata to store in the cache, for filtering purposes.
                This must be JSON serializable in an Elasticsearch document. Default to None.
        """
        self._es_client = es_client
        self._es_index = es_index
        self._store_input = store_input
        self._store_timestamp = store_timestamp
        self._namespace = namespace
        self._metadata = metadata
        self._manage_index()

    def _manage_index(self):
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
    def mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {
            "mappings": {
                "properties": {
                    "llm_input": {
                        "type": "text",
                        "index_prefixes": {"min_chars": 1, "max_chars": 10},
                    },
                    "vector_dump": {"type": "keyword", "index": False},
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"},
                }
            }
        }

    @staticmethod
    def _key(namespace: str, input_text: str) -> str:
        """Generate a key for the store."""
        return hashlib.md5((namespace + input_text).encode()).hexdigest()

    def mget(self, keys: Sequence[str]) -> List[Optional[List[float]]]:
        """Get the values associated with the given keys."""
        return []

    def mset(self, key_value_pairs: Sequence[Tuple[str, List[float]]]) -> None:
        """Set the values for the given keys."""
        return

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values."""
        return

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix."""
        yield ""
