# llm-elasticsearch-cache

A caching layer for LLMs that exploits Elasticsearch, fully compatible with Langchain caching.

## Install

```shell
pip install llm-elasticsearch-cache
```

## Usage

The Langchain cache can be used similarly to the
[other cache integrations](https://python.langchain.com/docs/integrations/llms/llm_caching).

Basic example

```python
from langchain.globals import set_llm_cache
from llmescache.langchain import ElasticsearchCache
from elasticsearch import Elasticsearch

es_client = Elasticsearch(hosts="localhost:9200")
set_llm_cache(ElasticsearchCache(es_client=es_client, es_index="llm-langchain-cache"))
```

The `es_index` parameter can also take aliases. This allows to use the 
[ILM: Manage the index lifecycle](https://www.elastic.co/guide/en/elasticsearch/reference/current/index-lifecycle-management.html)
that we suggest to consider for managing retention and controlling cache growth.
