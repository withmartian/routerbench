from typing import Optional

import numpy as np
import pandas as pd
import tqdm
from numpy.typing import NDArray
from sklearn.svm import LinearSVR

from embedding.cache import EmbeddingCache
from routers.abstract_router import (AbstractRouter, get_completion_token_cost,
                                     get_model_request_cost,
                                     get_prompt_token_cost,
                                     get_tokens_for_response)


class SVMRouter(AbstractRouter):
    def __init__(
        self,
        embedding_model: str,
        cache_url: Optional[str] = None,
        cache: Optional[EmbeddingCache] = None,
        train_file: pd.DataFrame | str = "data_path.pkl",
        models_to_route: list[str] = (" ",),
        **kwargs,
    ) -> None:
        self.models_to_route = models_to_route
        if isinstance(train_file, str):
            self.train_data = pd.read_pickle(train_file)
        else:
            self.train_data = train_file
        if cache:
            self.cache = cache
        else:
            self.cache = EmbeddingCache(cache_url)
        self.embedding_model = embedding_model
        embeddings_to_fit = self.cache.batch_get_embedding(
            tuple(self.train_data.prompt.values.tolist()), embedding_model
        )
        # Fit multiple MLPs, one for each model
        self.svrs = {}
        for model in tqdm.tqdm(models_to_route):
            self.svrs[model] = LinearSVR(random_state=1234)
            non_nan_idxs = self.train_data[model].notna()
            input_embeddings = np.vstack(embeddings_to_fit)[non_nan_idxs]
            performance_values = self.train_data[model].values[non_nan_idxs]
            self.svrs[model].fit(
                input_embeddings,
                performance_values,  # Performance value, should only take ones that are not NaN, same with the embeddings
            )
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
        prompt_embedding = self.cache.batch_get_embedding(
            tuple(prompts), self.embedding_model
        )
        prompt_embedding = prompt_embedding.tolist()
        performance_scores: dict = self.return_optimal_model(prompt_embedding)
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
        # Get key of the max value for each of the inputs in the lists
        # Want idxmax to return the model name, not the performance score
        best_model = pd.DataFrame(performance_scores).idxmax(axis=1)
        return best_model.values

    def get_model_to_route_to(self, input_embedding):
        # Score each model
        model_scores = {}
        for model in self.svrs:
            model_scores[model] = self.svrs[model].predict(input_embedding)
        return model_scores

    def return_optimal_model(self, input_embedding) -> dict:
        # Currently it only does the name of the model the router predicts
        models_to_route_to = self.get_model_to_route_to(input_embedding)
        return models_to_route_to
