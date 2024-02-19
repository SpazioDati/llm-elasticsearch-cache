from typing import List, Optional, Union, Iterator, Sequence, Tuple

from langchain_core.stores import BaseStore, K, V


class ElasticsearchStore(BaseStore[str, List[float]]):
    def mget(self, keys: Sequence[K]) -> List[Optional[V]]:
        return []

    def mset(self, key_value_pairs: Sequence[Tuple[K, V]]) -> None:
        return

    def mdelete(self, keys: Sequence[K]) -> None:
        return

    def yield_keys(
        self, *, prefix: Optional[str] = None
    ) -> Union[Iterator[K], Iterator[str]]:
        yield ""
