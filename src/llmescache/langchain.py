import hashlib
from datetime import datetime
from typing import Any, Optional, Dict

import elasticsearch
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.load import dumps


class ElasticSearchCache(BaseCache):
    """Cache store for LLM using ElasticSearch."""

    def __init__(
        self,
        es_client: elasticsearch.Elasticsearch,
        es_index: str,
        store_input: bool = True,
        store_datetime: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ElasticSearch cache store.

        Parameters:
            es_client: Elasticsearch
                The Elasticsearch client to use for the cache store.
            es_index: str
                The name of the index to use for the cache store.
            store_input: bool
                Whether to store the LLM input in the cache, i.e. the input prompt.
            store_datetime: bool
                Whether to store the datetime in the cache, i.e. the time of the first request for an input.
            metadata: Optional[dict]
                Additional metadata to store in the cache, i.e for filtering. Must be JSON serializable.
        """

        self._es_client = es_client
        self.index = es_index

        self.store_input = store_input
        self.store_datetime = store_datetime
        self.metadata = metadata or {}

        if not self._es_client.ping():
            raise elasticsearch.exceptions.ConnectionError(
                "ElasticSearch cluster is not available, not able to set up the cache store."
            )

        if not self._es_client.indices.exists(index=self.index):
            raise elasticsearch.exceptions.ConnectionError(
                f"ElasticSearch index {self.index} does not exist"
            )

    @staticmethod
    def _key(prompt: str, llm_string: str) -> str:
        def _hash(_input: str) -> str:
            return hashlib.md5(_input.encode()).hexdigest()

        return _hash(prompt + llm_string)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        try:
            record = self._es_client.get(
                index=self.index, id=self._key(prompt, llm_string)
            )
            return record["_source"]["llm_output"]
        except elasticsearch.exceptions.NotFoundError:
            return None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        body = {
            "llm_params": llm_string,
            "llm_output": dumps(return_val),
        }

        if self.metadata:
            body["metadata"] = self.metadata

        if self.store_input:
            body["llm_input"] = prompt

        if self.store_datetime:
            body["date"] = datetime.now().isoformat()

        self._es_client.index(
            index=self.index, id=self._key(prompt, llm_string), body=body
        )

    def clear(self, **kwargs: Any) -> None:
        raise NotImplementedError()
