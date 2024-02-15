import pytest
from unittest.mock import MagicMock, patch

from elasticsearch import Elasticsearch

from llmescache.langchain import ElasticsearchCache


@pytest.fixture
def es_client_mock():
    with patch.object(Elasticsearch, "__init__", return_value=None):
        client_mock = MagicMock(spec=Elasticsearch)
        client_mock.indices = MagicMock()
        yield client_mock


@pytest.fixture
def cache(es_client_mock):
    yield ElasticsearchCache(
        es_client=es_client_mock,
        es_index="test_index",
        store_input=True,
        store_timestamp=True,
        store_input_params=True,
        metadata={"test": "metadata"}
    )
