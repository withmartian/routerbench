from typing import List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from embedding.cache import EmbeddingCache
from routers.abstract_router import (AbstractRouter, get_completion_token_cost,
                                     get_model_request_cost,
                                     get_prompt_token_cost,
                                     get_tokens_for_response)


class KNNRouter(AbstractRouter):
    def __init__(
        self,
        embedding_model: str,
        cache_url: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        train_file: pd.DataFrame | str = "data_path.pkl",
        n_neighbors: int = 50,
        distance_metric: str = "cosine",
        leaf_size: int = 30,
        models_to_route: list[str] = (" ",),
        **kwargs,
    ) -> None:
        self.models_to_route = models_to_route
        from sklearn.neighbors import NearestNeighbors

        if isinstance(train_file, str):
            self.train_data = pd.read_pickle(train_file)
        else:
            self.train_data = train_file

        # Call the embedding cache to get the embeddings
        if cache is not None:
            self.cache = cache
        else:
            self.cache = EmbeddingCache(cache_url)
        self.embedding_model = embedding_model
        embeddings_to_fit = self.cache.batch_get_embedding(
            tuple(self.train_data.prompt.values.tolist()), self.embedding_model
        )
        self.knn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=distance_metric,
            leaf_size=leaf_size,
            n_jobs=-1,
        ).fit(np.vstack(embeddings_to_fit))
        self.average_response_length = (
            self.calculate_average_response_length_per_model()
        )

    def calculate_average_response_length_per_model(self) -> dict:
        average_response_length = {}
        for model in self.models_to_route:
            average_response_length[model] = (
                self.train_data[f"{model}|model_response"]
                .apply(lambda x: get_tokens_for_response(x, model))
                .mean()
            )
        return average_response_length

    def batch_route_prompts(self, prompts: list[str], **kwargs) -> NDArray[str]:
        """

        :param prompts: List of prompts to route
        :param kwargs:
        :return:
        """
        prompt_embeddings = self.cache.batch_get_embedding(
            tuple(prompts), self.embedding_model
        )
        if len(prompt_embeddings.shape) == 1:
            prompt_embeddings = np.vstack(prompt_embeddings)
        performance_scores = self.calc_performance_scores(prompt_embeddings)

        if "willingness_to_pay" in kwargs and kwargs["willingness_to_pay"] > 0:
            willingness_to_pay = kwargs["willingness_to_pay"]
            for model_name in self.models_to_route:
                performance_scores[model_name] = performance_scores[
                    model_name
                ] * willingness_to_pay - (
                    get_completion_token_cost(model_name)
                    + (
                        get_model_request_cost(model_name)
                        / self.average_response_length[model_name]
                    )
                    + get_prompt_token_cost(model_name)
                )
        models_to_route_to = performance_scores.idxmax(axis=1)

        return models_to_route_to.values

    def calc_performance_scores(self, prompt_embeddings: np.ndarray) -> pd.DataFrame:
        """

        :param prompt_embeddings: first dim is batch size, second dim is embedding size.
        :param kwargs:
        :return:
        """
        _, indices = self.knn.kneighbors(prompt_embeddings)

        indices_df = pd.DataFrame(indices)
        top_k_rows = self.train_data.iloc[indices_df.values.flatten()]
        top_k_rows = top_k_rows.assign(
            group=np.repeat(indices_df.index, indices_df.shape[1])
        )
        performance_scores = top_k_rows.groupby("group")[self.models_to_route].mean()
        return performance_scores
