from abc import ABC, abstractmethod

import pandas as pd

from embedding import EmbeddingCache


class AbstractConvertor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def convert(self, input_data: pd.DataFrame) -> pd.DataFrame:
        pass

    def convert_and_embed(
        self,
        input_data: pd.DataFrame,
        cache: EmbeddingCache,
        embedding_models: list[str],
    ) -> pd.DataFrame:
        converted_dataframe = self.convert(input_data)
        # Run embeddings here
        for embedding_model in embedding_models:
            cache.batch_get_embedding(
                tuple(converted_dataframe.prompt.values),
                embedding_model=embedding_model,
            )
        return converted_dataframe
