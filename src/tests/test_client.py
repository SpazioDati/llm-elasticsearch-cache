import pytest

from elasticsearch import exceptions


from llmescache.langchain import ElasticSearchCache


def test_initialization_failure(es_client_mock):
    es_client_mock.ping.return_value = False
    with pytest.raises(exceptions.ConnectionError):
        ElasticSearchCache(es_client=es_client_mock, es_index="test_index")


def test_initialization_success(es_client_mock):
    es_client_mock.ping.return_value = True
    es_client_mock.indices.exists.return_value = False
    cache = ElasticSearchCache(es_client=es_client_mock, es_index="test_index")  # noqa
    es_client_mock.indices.create.assert_called_once()


def test_key_generation(cache):
    prompt = "test_prompt"
    llm_string = "test_llm_string"
    key = cache._key(prompt, llm_string)
    assert isinstance(key, str) and len(key) > 0


def test_clear(cache, es_client_mock):
    cache.clear()
    es_client_mock.delete_by_query.assert_called_once_with(
        index="test_index",
        body={"query": {"match_all": {}}},
        refresh=True,
        wait_for_completion=True,
    )
