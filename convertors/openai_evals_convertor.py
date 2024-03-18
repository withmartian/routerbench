import pandas as pd

from convertors.abstract_convertor import AbstractConvertor
from convertors.common import (calculate_total_cost_for_prompts_and_responses,
                               convert_to_wide_format,
                               generate_oracle_routing_results)


class OpenAIEvalsConvertor(AbstractConvertor):
    def __init__(self, **kwargs):
        super().__init__()

    def convert(self, input_data: pd.DataFrame) -> pd.DataFrame:
        models_to_route = input_data.model_name.unique().tolist()
        print(models_to_route)
        # Check file extension and load the file accordingly
        input_data = convert_to_wide_format(input_data)
        print(input_data.columns)
        input_data = calculate_total_cost_for_prompts_and_responses(
            input_data, models_to_route=models_to_route
        )
        input_data = generate_oracle_routing_results(
            input_data, model_to_route=models_to_route
        )
        input_data["eval_name"] = input_data["sample_id"].apply(
            lambda name: name.split(".")[0]
        )
        return input_data
