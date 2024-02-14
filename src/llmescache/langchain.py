import hashlib
from datetime import datetime
from typing import Any, Dict, Optional

import elasticsearch
from langchain_community.cache import _dumps_generations, _loads_generations
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache


class ElasticSearchCache(BaseCache):
    """Cache store for LLM using ElasticSearch."""

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
        Initialize the ElasticSearch cache store by specifying the index
        to use and determining what additional information (like input, timestamp, input parameters,
        and any other metadata) should be stored in the cache.

        Args:
            es_client (Elasticsearch): The Elasticsearch client to use for the cache store.
            es_index (str): The name of the index to use for the cache store. It will be created if it does not exist
                according to the default mapping defined by `mapping` property.
            store_input (bool): Whether to store the LLM input in the cache, i.e., the input prompt. Default to True.
            store_timestamp (bool): Whether to store the datetime in the cache, i.e., the time of the
                first request for a LLM input. Default to True.
            store_input_params (bool): Whether to store the input parameters in the cache, i.e., the LLM
                parameters used to generate the LLM response. Default to True.
            metadata (Optional[dict], optional): Additional metadata to store in the cache, for filtering purposes.
                This must be JSON serializable in an ElasticSearch document. Default to None.
        """

        self._es_client = es_client
        self._es_index = es_index

        self._store_input = store_input
        self._store_timestamp = store_timestamp
        self._store_input_params = store_input_params
        self._metadata = metadata

        if not self._es_client.ping():
            raise elasticsearch.exceptions.ConnectionError(
                "ElasticSearch cluster is not available, not able to set up the cache store."
            )

        if not self._es_client.indices.exists(index=self._es_index):
            self._es_client.indices.create(index=self._es_index, body=self.mapping)

    @property
    def mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "llm_output": {
                        "type": "text",
                        "index": "false"
                    },
                    "llm_params": {
                        "type": "text",
                        "index": "false"
                    },
                    "llm_input": {
                        "type": "text",
                        "index": "false"
                    },
                    "metadata": {
                        "dynamic": "true",
                        "type": "object"
                    },
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
        try:
            record = self._es_client.get(
                index=self._es_index, id=self._key(prompt, llm_string)
            )
            return _loads_generations(record["_source"]["llm_output"])
        except elasticsearch.exceptions.NotFoundError:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        body = {
            "llm_output": _dumps_generations(return_val),
        }

        if self._store_input_params:
            body["llm_params"] = llm_string

        if self._metadata is not None:
            body["metadata"] = self.metadata  # type: ignore

        if self._store_input:
            body["llm_input"] = prompt

        if self._store_timestamp:
            body["timestamp"] = datetime.now().isoformat()

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
