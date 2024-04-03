from datetime import datetime

from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import NotFoundError
from langchain_core.load import dumps
from langchain_core.outputs import Generation


def test_key_generation(es_cache_fx):
    key1 = es_cache_fx._key("test_prompt", "test_llm_string")
    assert key1 and isinstance(key1, str)
    key2 = es_cache_fx._key("test_prompt", "test_llm_string1")
    assert key2 and key1 != key2
    key3 = es_cache_fx._key("test_prompt1", "test_llm_string")
    assert key3 and key1 != key3


def test_clear(es_cache_fx):
    es_cache_fx.clear()
    es_cache_fx._es_client.delete_by_query.assert_called_once_with(
        index="test_index",
        body={"query": {"match_all": {}}},
        refresh=True,
        wait_for_completion=True,
    )


def test_build_document(es_cache_fx):
    doc = es_cache_fx.build_document(
        "test_prompt", "test_llm_string", [Generation(text="test_prompt")]
    )
    assert doc["llm_input"] == "test_prompt"
    assert doc["llm_params"] == "test_llm_string"
    assert isinstance(doc["llm_output"], list)
    assert all(isinstance(gen, str) for gen in doc["llm_output"])
    assert datetime.fromisoformat(str(doc["timestamp"]))
    assert doc["metadata"] == es_cache_fx._metadata


def test_update(es_cache_fx):
    es_cache_fx.update("test_prompt", "test_llm_string", [Generation(text="test")])
    timestamp = es_cache_fx._es_client.index.call_args.kwargs["body"]["timestamp"]
    doc = es_cache_fx.build_document(
        "test_prompt", "test_llm_string", [Generation(text="test")]
    )
    doc["timestamp"] = timestamp
    es_cache_fx._es_client.index.assert_called_once_with(
        index=es_cache_fx._es_index,
        id=es_cache_fx._key("test_prompt", "test_llm_string"),
        body=doc,
        require_alias=es_cache_fx._is_alias,
        refresh=True,
    )


def test_lookup(es_cache_fx):
    cache_key = es_cache_fx._key("test_prompt", "test_llm_string")
    doc = {"_source": {"llm_output": [dumps(Generation(text="test"))]}}
    es_cache_fx._is_alias = False
    es_cache_fx._es_client.get.side_effect = NotFoundError(
        "not found",
        ApiResponseMeta(404, "0", HttpHeaders(), 0, NodeConfig("http", "xxx", 80)),
        "",
    )
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") is None
    es_cache_fx._es_client.get.assert_called_once_with(
        index="test_index", id=cache_key, source=["llm_output"]
    )
    es_cache_fx._es_client.get.side_effect = None
    es_cache_fx._es_client.get.return_value = doc
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") == [
        Generation(text="test")
    ]
    es_cache_fx._is_alias = True
    es_cache_fx._es_client.search.return_value = {
        "hits": {"total": {"value": 0}, "hits": []}
    }
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") is None
    es_cache_fx._es_client.search.assert_called_once_with(
        index="test_index",
        body={"query": {"term": {"_id": cache_key}}},
        source_includes=["llm_output"],
    )
    doc["_index"] = "index_1"
    doc2 = {
        "_index": "index_2",
        "_source": {"llm_output": [dumps(Generation(text="test2"))]},
    }
    es_cache_fx._es_client.search.return_value = {
        "hits": {"total": {"value": 2}, "hits": [doc, doc2]}
    }
    assert es_cache_fx.lookup("test_prompt", "test_llm_string") == [
        Generation(text="test2")
    ]
