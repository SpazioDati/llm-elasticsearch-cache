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
        store_datetime: bool = True,
        store_input_param: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ElasticSearch cache store.

        This method initializes an object with ElasticSearch caching capabilities.
        It configures the cache store using an Elasticsearch client, specifying the index
        to use, and determining what additional information (like input, datetime, input parameters,
        and any other metadata) should be stored in the cache.

        Args:
            es_client (Elasticsearch): The Elasticsearch client to use for the cache store.
            es_index (str): The name of the index to use for the cache store.
            store_input (bool): Whether to store the LLM input in the cache, i.e., the input prompt.
                Defaults to True.
            store_datetime (bool): Whether to store the datetime in the cache, i.e., the time of the
                first request for an input. Defaults to True.
            store_input_param (bool): Whether to store the input parameters in the cache, i.e., the
                parameters used to generate the LLM input. Defaults to True.
            metadata (Optional[dict], optional): Additional metadata to store in the cache, for filtering purposes.
                This must be JSON serializable. Defaults to None.
        """

        self._es_client = es_client
        self.index = es_index

        self.store_input = store_input
        self.store_datetime = store_datetime
        self.store_input_param = store_input_param
        self.metadata = metadata or {}

        if not self._es_client.ping():
            raise elasticsearch.exceptions.ConnectionError(
                "ElasticSearch cluster is not available, not able to set up the cache store."
            )

        if not self._es_client.indices.exists(index=self.index):
            self._es_client.indices.create(
                index=self.index, body=self.get_default_mapping
            )

    @property
    def get_default_mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {
            "mappings": {
                "properties": {
                    "llm_output": {"type": "text"},
                    "llm_params": {"type": "text"},
                    "llm_input": {"type": "text"},
                    "metadata": {"type": "object"},
                    "date": {"type": "date"},
                }
            }
        }

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        """Generate a key for the cache store."""

        def _hash(_input: str) -> str:
            return hashlib.md5(_input.encode()).hexdigest()

        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        try:
            record = self._es_client.get(
                index=self.index, id=self._key(prompt, llm_string)
            )
            return _loads_generations(record["_source"]["llm_output"])
        except elasticsearch.exceptions.NotFoundError:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update based on prompt and llm_string."""
        body = {
            "llm_output": _dumps_generations(return_val),
        }

        if self.store_input_param:
            body["llm_params"] = llm_string

        if self.metadata:
            body["metadata"] = self.metadata  # type: ignore

        if self.store_input:
            body["llm_input"] = prompt

        if self.store_datetime:
            body["date"] = datetime.now().isoformat()

        self._es_client.index(
            index=self.index, id=self._key(prompt, llm_string), body=body
        )

    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self._es_client.delete_by_query(
            index=self.index,
            body={"query": {"match_all": {}}},
            refresh=True,
            wait_for_completion=True,
        )
