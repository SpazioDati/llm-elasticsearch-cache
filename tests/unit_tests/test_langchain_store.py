from datetime import datetime


def test_key_generation(es_store_fx):
    key1 = es_store_fx._key("test_text")
    assert key1 and isinstance(key1, str)
    key2 = es_store_fx._key("test_text2")
    assert key2 and key1 != key2
    es_store_fx._namespace = "other"
    key3 = es_store_fx._key("test_text")
    assert key3 and key1 != key3
    es_store_fx._namespace = None
    key4 = es_store_fx._key("test_text")
    assert key4 and key1 != key4 and key3 != key4


def test_build_document(es_store_fx):
    doc = es_store_fx.build_document("test_text", [1.5, 2, 3.6])
    assert doc["llm_input"] == "test_text"
    assert doc["vector_dump"] == [1.5, 2, 3.6]
    assert datetime.fromisoformat(str(doc["timestamp"]))
    assert doc["metadata"] == es_store_fx._metadata


def test_mget(es_store_fx):
    cache_keys = [
        es_store_fx._key("test_text1"),
        es_store_fx._key("test_text2"),
        es_store_fx._key("test_text3"),
    ]
    docs = {
        "docs": [
            {"_index": "test_index", "_id": cache_keys[0], "found": False},
            {
                "_index": "test_index",
                "_id": cache_keys[1],
                "found": True,
                "_source": {"vector_dump": [1.5, 2, 3.6]},
            },
            {
                "_index": "test_index",
                "_id": cache_keys[2],
                "found": True,
                "_source": {"vector_dump": [5, 6, 7.1]},
            },
        ]
    }
    es_store_fx._is_alias = False
    es_store_fx._es_client.mget.return_value = docs
    assert es_store_fx.mget([]) == []
    assert es_store_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        [1.5, 2, 3.6],
        [5, 6, 7.1],
    ]
    es_store_fx._es_client.mget.assert_called_with(
        index="test_index", ids=cache_keys, source_includes=["vector_dump"]
    )
    es_store_fx._is_alias = True
    es_store_fx._es_client.search.return_value = {
        "hits": {"total": {"value": 0}, "hits": []}
    }
    assert es_store_fx.mget([]) == []
    assert es_store_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        None,
        None,
    ]
    es_store_fx._es_client.search.assert_called_with(
        index="test_index",
        body={
            "query": {"ids": {"values": cache_keys}},
            "size": 3,
        },
        source_includes=["vector_dump"],
    )
    resp = {
        "hits": {"total": {"value": 3}, "hits": [d for d in docs["docs"] if d["found"]]}
    }
    es_store_fx._es_client.search.return_value = resp
    assert es_store_fx.mget(["test_text1", "test_text2", "test_text3"]) == [
        None,
        [1.5, 2, 3.6],
        [5, 6, 7.1],
    ]
