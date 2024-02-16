import hashlib
from datetime import datetime
from operator import itemgetter
from typing import Any, Optional, Dict
import elasticsearch
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps, loads


class ElasticsearchCache(BaseCache):
    """Cache store for LLM using Elasticsearch."""

    def __init__(
        self,
        es_client: elasticsearch.Elasticsearch,
        es_index: str,
        store_input: bool = True,
        store_timestamp: bool = True,
        store_input_params: bool = True,
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
            store_input_params (bool): Whether to store the input parameters in the cache, i.e., the LLM
                parameters used to generate the LLM response. Default to True.
            metadata (Optional[dict], optional): Additional metadata to store in the cache, for filtering purposes.
                This must be JSON serializable in an Elasticsearch document. Default to None.
        """

        self._es_client = es_client
        self._es_index = es_index
        self._store_input = store_input
        self._store_timestamp = store_timestamp
        self._store_input_params = store_input_params
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
                    "llm_output": {"type": "text", "index": False},
                    "llm_params": {"type": "text", "index": False},
                    "llm_input": {"type": "text", "index": False},
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"},
                }
            }
        }

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        """Generate a key for the cache store."""
        return hashlib.md5((prompt + llm_string).encode()).hexdigest()

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        cache_key = self._key(prompt, llm_string)
        if self._is_alias:
            result = self._es_client.search(
                index=self._es_index,
                body={"query": {"term": {"_id": cache_key}}},
                source_includes=["llm_output"],
            )
            if result["hits"]["total"]["value"] > 0:
                # get the record from the latest index, assuming lexicographic order is chronological
                record = max(result["hits"]["hits"], key=itemgetter("_index"))
            else:
                return None
        else:
            try:
                record = self._es_client.get(
                    index=self._es_index, id=cache_key, source=["llm_output"]
                )
            except elasticsearch.exceptions.NotFoundError:
                return None
        return [loads(item) for item in record["_source"]["llm_output"]]

    def build_document(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> Dict[str, Any]:
        """Build the Elasticsearch document for storing a single LLM interaction"""
        body: Dict[str, Any] = {
            "llm_output": [dumps(item) for item in return_val],
        }
        if self._store_input_params:
            body["llm_params"] = llm_string
        if self._metadata is not None:
            body["metadata"] = self._metadata
        if self._store_input:
            body["llm_input"] = prompt
        if self._store_timestamp:
            body["timestamp"] = datetime.now().isoformat()
        return body

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        body = self.build_document(prompt, llm_string, return_val)
        self._es_client.index(
            index=self._es_index, id=self._key(prompt, llm_string), body=body
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._es_client.delete_by_query(
            index=self._es_index,
            body={"query": {"match_all": {}}},
            refresh=True,
            wait_for_completion=True,
        )
