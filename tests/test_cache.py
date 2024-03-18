import mongomock

from embedding.cache import EmbeddingCache


@mongomock.patch(servers=(("server.example.com", 27017),))
def test_embedding_cache():
    cache = EmbeddingCache("server.example.com")
    prompts = ["test1", "test2", "test3"]
    embedding_model = "all-MiniLM-L12-v2"
    embeddings = cache.batch_get_embedding(prompts, embedding_model)
    assert len(embeddings) == len(prompts)
