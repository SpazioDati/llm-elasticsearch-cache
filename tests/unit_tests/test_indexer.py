import pytest
from elasticsearch import exceptions

from llmescache.langchain import ElasticsearchCache, ElasticsearchStore


@pytest.mark.parametrize("implementation", [ElasticsearchCache, ElasticsearchStore])
def test_initialization(es_client_fx, implementation):
    es_client_fx.ping.return_value = False
    with pytest.raises(exceptions.ConnectionError):
        implementation(es_client=es_client_fx, es_index="test_index")
    es_client_fx.ping.return_value = True
    es_client_fx.indices.exists_alias.return_value = True
    cache = implementation(es_client=es_client_fx, es_index="test_index")
    cache._es_client.indices.exists_alias.assert_called_with(name="test_index")
    assert cache._is_alias
    cache._es_client.indices.put_mapping.assert_called_with(
        index="test_index", body=cache.mapping["mappings"]
    )
    es_client_fx.indices.exists_alias.return_value = False
    es_client_fx.indices.exists.return_value = False
    cache = implementation(es_client=es_client_fx, es_index="test_index")
    assert not cache._is_alias
    cache._es_client.indices.create.assert_called_with(
        index="test_index", body=cache.mapping
    )


@pytest.mark.parametrize("indexer", ["es_cache_fx", "es_store_fx"])
def test_mapping(indexer, request):
    mapping = request.getfixturevalue(indexer).mapping
    assert mapping.get("mappings")
    assert mapping["mappings"].get("properties")
