import pytest
from unittest.mock import MagicMock
from elasticsearch import Elasticsearch
from elasticsearch._sync.client import IndicesClient

from llmescache.langchain import ElasticsearchCache


@pytest.fixture
def es_client_fx():
    client_mock = MagicMock(spec=Elasticsearch)
    client_mock.indices = MagicMock(spec=IndicesClient)
    yield client_mock()


@pytest.fixture
def es_cache_fx(es_client_fx):
    yield ElasticsearchCache(
        es_client=es_client_fx,
        es_index="test_index",
        store_input=True,
        store_timestamp=True,
        store_input_params=True,
        metadata={"project": "test"}
    )
