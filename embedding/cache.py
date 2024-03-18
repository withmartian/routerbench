import hashlib
import logging
import os
import pickle
from functools import lru_cache
from typing import Optional

import fsspec
import numpy as np
import pymongo
import sentence_transformers
from dotenv import load_dotenv
from tqdm import tqdm

_log = logging.getLogger(__name__)


def save_cache_to_disk(results, path: str) -> None:
    with fsspec.open(path, "wb") as f:
        # save results as pickle
        f.write(pickle.dumps(results))
    return None


class EmbeddingCache:
    def __init__(
        self,
        connection_string: str = None,
        local_cache_path: Optional[str] = None,
        local_mode: bool = False,
    ):
        self.local_mode = local_mode
        if not local_mode:
            self.client = pymongo.MongoClient(connection_string)
            self.db = self.client["embedding_cache"]
        self.local_cache = []
        self.local_cache_path = local_cache_path

    @lru_cache(maxsize=1000)
    def batch_get_embedding(self, prompts: tuple, embedding_model: str) -> np.ndarray:
        """
        Get the embeddings for a list of prompts

        Prompts have to be a tuple for LRU cache to work

        :param prompts: Prompts to get the embeddings for, as a tuple of strings
        :param embedding_model: Embedding model name to use
        :return: Embeddings for the prompts
        """
        # Deduplicate the prompts
        print(f"Getting embeddings for {len(prompts)} prompts")
        hashes = [self._compute_hash(prompt, embedding_model) for prompt in prompts]
        if self.local_cache:
            local_hashes = [result["hash"] for result in self.local_cache]
            if all([h in local_hashes for h in hashes]):
                # Get the embeddings from the local cache
                embeddings = np.asarray(
                    [
                        self.local_cache[local_hashes.index(h)]["embedding"]
                        for h in hashes
                    ]
                )
                print(f"Got {len(embeddings)} embeddings from the local cache list")
                return embeddings
        if self.local_cache_path is not None and os.path.exists(self.local_cache_path):
            with fsspec.open(self.local_cache_path, "rb") as f:
                self.local_cache = pickle.loads(f.read())
            # Check if all hashes are in local cache
            local_hashes = [result["hash"] for result in self.local_cache]
            if all([h in local_hashes for h in hashes]):
                # Get the embeddings from the local cache
                embeddings = np.asarray(
                    [
                        self.local_cache[local_hashes.index(h)]["embedding"]
                        for h in hashes
                    ]
                )
                print(
                    f"Got {len(embeddings)} embeddings from the local cache file {self.local_cache_path}"
                )
                return embeddings
        if os.path.exists(f"data/embedding_cache_{embedding_model}.pkl"):
            with fsspec.open(f"data/embedding_cache_{embedding_model}.pkl", "rb") as f:
                self.local_cache = pickle.loads(f.read())
            # Check if all hashes are in local cache
            local_hashes = [result["hash"] for result in self.local_cache]
            if all([h in local_hashes for h in hashes]):
                # Get the embeddings from the local cache
                embeddings = np.asarray(
                    [
                        self.local_cache[local_hashes.index(h)]["embedding"]
                        for h in hashes
                    ]
                )
                print(f"Got {len(embeddings)} embeddings from the local cache")
                return embeddings
        # Check if the hash is already in the local cache
        if self.local_mode:
            embeddings = self._compute_embedding(prompts, embedding_model)
            hashes = [self._compute_hash(prompt, embedding_model) for prompt in prompts]
            results = [
                {
                    "prompt": prompt,
                    "embedding": embeddings[i],
                    "hash": hashes[i],
                    "embedding_model": embedding_model,
                }
                for i, prompt in enumerate(prompts)
            ]
            # Add to local cache
        else:
            # Build up a list of hashes that are not in the local cache
            print(f"Checking if {len(hashes)} prompts are in the cache")
            # Do it in batches of 5000 hashes at a time
            results = []
            search_hashes = list(set(hashes))
            for i in tqdm(range(0, len(search_hashes), 10000)):
                results += list(
                    self.db.embedding_cache.find(
                        {"hash": {"$in": search_hashes[i : i + 10000]}}
                    )
                )
            _log.info(
                f"Got {len(results)} cached embeddings for {len(prompts)} prompts from the cache"
            )
            # Get all items in self.collection that has a hash in hashes
            # If the hash is not in the collection, run the embedding model and save it
            if len(results) != len(prompts):
                self.batch_add_embedding(prompts, embedding_model)
                results = []
                for i in tqdm(range(0, len(hashes), 10000)):
                    results += list(
                        self.db.embedding_cache.find(
                            {"hash": {"$in": search_hashes[i : i + 10000]}}
                        )
                    )
        if os.path.exists(f"data/embedding_cache_{embedding_model}.pkl"):
            with fsspec.open(f"data/embedding_cache_{embedding_model}.pkl", "rb") as f:
                self.local_cache = pickle.loads(f.read())
            self.local_cache += results

            save_cache_to_disk(
                self.local_cache, f"data/embedding_cache_{embedding_model}.pkl"
            )
        else:
            save_cache_to_disk(results, f"data/embedding_cache_{embedding_model}.pkl")
        # Incase there are duplicates, need to add them in to the results, in the same order
        result_hashes = [result["hash"] for result in results]
        hash_indicies = [result_hashes.index(h) for h in hashes]
        final_results = []
        for idx in hash_indicies:
            final_results.append(results[idx])
        # Return the list of embeddings
        embeddings = np.asarray([result["embedding"] for result in final_results])
        return embeddings

    def batch_add_embedding(self, prompts: tuple, embedding_model: str) -> None:
        """
        Add the embeddings for a list of prompts to the cache

        :param prompts: Prompts to add the embeddings for, as a tuple of strings
        :param embedding_model: Embedding model to use
        """
        # Check if the hash is already in the collection
        prompts = list(set(prompts))
        hashes = [self._compute_hash(prompt, embedding_model) for prompt in prompts]
        results = []
        for i in tqdm(range(0, len(hashes), 10000)):
            results += list(
                self.db.embedding_cache.find({"hash": {"$in": hashes[i : i + 10000]}})
            )
        cached_hashes = [result["hash"] for result in results]
        # Check which prompts are not in the results
        prompts = [
            prompt
            for prompt in prompts
            if self._compute_hash(prompt, embedding_model) not in cached_hashes
        ]
        if len(prompts) == 0:
            return
        _log.info(
            f"Adding embeddings for {len(prompts)} prompts to the cache, after filtering"
        )
        # Compute the embeddings for the prompts that are not in the collection
        embeddings = self._compute_embedding(prompts, embedding_model)
        hashes = [self._compute_hash(prompt, embedding_model) for prompt in prompts]
        items_to_insert = []
        for i, prompt in tqdm(enumerate(prompts)):
            items_to_insert.append(
                {
                    "prompt": prompt,
                    "embedding": embeddings[i],
                    "hash": hashes[i],
                    "embedding_model": embedding_model,
                }
            )
            if len(items_to_insert) == 1000:
                self.db.embedding_cache.insert_many(items_to_insert)
                items_to_insert = []
        if len(items_to_insert) > 0:
            self.db.embedding_cache.insert_many(items_to_insert)
        _log.info(f"Added embeddings for {len(items_to_insert)} prompts to the cache")
        return None

    def _compute_embedding(self, prompts, embedding_model) -> list[float]:
        match embedding_model:
            case _:  # Not a special case, so use SentenceTransformers
                return self._compute_sentence_transformers_embedding(
                    prompts, embedding_model
                )

    @staticmethod
    def _compute_hash(prompt: str, embedding_model: str) -> str:
        return hashlib.sha256((prompt + embedding_model).encode()).hexdigest()

    @staticmethod
    def _compute_sentence_transformers_embedding(
        prompts: list[str], embedding_model: str
    ) -> list[float]:
        model = sentence_transformers.SentenceTransformer(embedding_model)
        return model.encode(prompts).tolist()
