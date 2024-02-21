import hashlib
import json
from datetime import datetime
from typing import List, Optional, Iterator, Sequence, Tuple, Any, Dict, Iterable

import elasticsearch
from elasticsearch import helpers
from langchain_core.stores import BaseStore

from llmescache.langchain.base import ElasticsearchIndexer


class ElasticsearchStore(BaseStore[str, List[float]], ElasticsearchIndexer):
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

    @property
    def mapping(self) -> Dict[str, Any]:
        """Get the default mapping for the index."""
        return {
            "mappings": {
                "properties": {
                    "llm_input": {"type": "text", "index": False},
                    "vector_dump": {"type": "keyword", "index": False},
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"},
                }
            }
        }

    def _key(self, input_text: str) -> str:
        """Generate a key for the store."""
        return hashlib.md5((self._namespace or "" + input_text).encode()).hexdigest()

    def mget(self, keys: Sequence[str]) -> List[Optional[List[float]]]:
        """Get the values associated with the given keys."""
        cache_keys = [self._key(k) for k in keys]
        if self._is_alias:
            results = self._es_client.search(
                index=self._es_index,
                body={
                    "query": {"ids": {"values": cache_keys}},
                    "size": len(cache_keys),
                },
                source_includes=["vector_dump"],
            )
            map_ids = {
                r["_id"]: json.loads(r["_source"]["vector_dump"])
                for r in results["hits"]["hits"]
            }
            return [map_ids.get(k) for k in cache_keys]
        else:
            records = self._es_client.mget(
                index=self._es_index, ids=cache_keys, source_includes=["vector_dump"]
            )
            return [
                json.loads(r["_source"]["vector_dump"]) if r["found"] else None
                for r in records["docs"]
            ]

    def build_document(self, llm_input: str, vector: List[float]) -> Dict[str, Any]:
        """Build the Elasticsearch document for storing a single embedding"""
        body: Dict[str, Any] = {"vector_dump": json.dumps(vector)}
        if self._metadata is not None:
            body["metadata"] = self._metadata
        if self._store_input:
            body["llm_input"] = llm_input
        if self._store_timestamp:
            body["timestamp"] = datetime.now().isoformat()
        return body

    def _bulk(self, actions: Iterable[Dict[str, Any]]):
        helpers.bulk(
            client=self._es_client,
            actions=actions,
            index=self._es_index,
            require_alias=self._is_alias,
            refresh=True,
        )

    def mset(self, key_value_pairs: Sequence[Tuple[str, List[float]]]) -> None:
        """Set the values for the given keys."""
        bodies = {
            self._key(pair[0]): self.build_document(*pair) for pair in key_value_pairs
        }
        actions = (
            {"_op_type": "index", "_id": key, "_source": doc}
            for key, doc in bodies.items()
        )
        self._bulk(actions)
        return

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values."""
        actions = ({"_op_type": "delete", "_id": self._key(key)} for key in keys)
        self._bulk(actions)
        return

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix."""
        # TODO This method is not currently used by CacheBackedEmbeddings, we can leave it blank.
        #      It could be implemented with ES "index_prefixes", but they are limited and expensive.
        raise NotImplementedError()