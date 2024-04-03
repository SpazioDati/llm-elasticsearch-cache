from unittest.mock import MagicMock
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.embeddings import FakeEmbeddings


def test_hit_and_miss(es_store_fx, fake_chat_fx):
    store_mock = MagicMock(es_store_fx)
    underlying_embeddings = FakeEmbeddings(size=3)
    cached_embeddings = CacheBackedEmbeddings(underlying_embeddings, store_mock)
    store_mock.mget.return_value = [None, None]
    assert all(cached_embeddings.embed_documents(["test_text1", "test_text2"]))
    store_mock.mget.assert_called_once()
    store_mock.mset.assert_called_once()
    store_mock.reset_mock()
    store_mock.mget.return_value = [None, [1.5, 2, 3.6]]
    assert all(cached_embeddings.embed_documents(["test_text1", "test_text2"]))
    store_mock.mget.assert_called_once()
    store_mock.mset.assert_called_once()
    assert len(store_mock.mset.call_args.args) == 1
    assert store_mock.mset.call_args.args[0][0][0] == "test_text1"
    store_mock.reset_mock()
    store_mock.mget.return_value = [[1.5, 2.3, 3], [1.5, 2, 3.6]]
    assert all(cached_embeddings.embed_documents(["test_text1", "test_text2"]))
    store_mock.mget.assert_called_once()
    store_mock.mset.assert_not_called()
