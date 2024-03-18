import ast
from abc import ABC, abstractmethod
from typing import Union

from numpy.typing import NDArray
from tokencost import count_string_tokens

from routers.common import TOKEN_COSTS


def get_prompt_token_cost(model_name: str) -> float:
    return TOKEN_COSTS[model_name]["prompt"]


def get_model_request_cost(model_name: str) -> float:
    try:
        return TOKEN_COSTS[model_name]["request"]
    except KeyError:
        return 0.0


def get_completion_token_cost(model_name: str) -> float:
    return TOKEN_COSTS[model_name]["completion"]


def get_tokens_for_response(response: Union[str, list[str]], model_name) -> int:
    if isinstance(response, str):
        if response[0] == "[" and response[-1] == "]":
            response = str(ast.literal_eval(response))
    elif not isinstance(response, list):
        response = str(response)  # want responses to be strings
    return count_string_tokens(response, model_name)


def calculate_cost_for_prompt_and_response(
    prompt: Union[str, list[str]], response: Union[str, list[str]], model_name: str
) -> float:
    """
    Calculate the cost in USD for a prompt given a model name

    :param prompt: The prompt to calculate the cost for
    :param model_name: The name of the model to calculate the cost for
    :return: The cost of the prompt
    """
    if isinstance(prompt, str):
        if prompt[0] == "[" and prompt[-1] == "]":
            prompt = ast.literal_eval(prompt)
    elif not isinstance(prompt, list):
        prompt = str(prompt)
    if isinstance(response, str):
        if response[0] == "[" and response[-1] == "]":
            response = ast.literal_eval(response)
    elif not isinstance(response, list):
        response = str(response)  # want responses to be strings
    if isinstance(prompt, list):
        prompt_cost = sum(
            [count_string_tokens(p, model_name) for p in prompt]
        ) * get_prompt_token_cost(model_name)
        response_cost = sum(
            [count_string_tokens(r, model_name) for r in response]
        ) * get_completion_token_cost(model_name)
        cost = prompt_cost + response_cost + get_model_request_cost(model_name)
    else:
        if isinstance(response, str):
            prompt_cost = count_string_tokens(
                prompt, model_name
            ) * get_prompt_token_cost(model_name)
            response_cost = count_string_tokens(
                response, model_name
            ) * get_completion_token_cost(model_name)
        else:
            tokens = count_string_tokens(prompt, model_name)
            prompt_cost = tokens * get_prompt_token_cost(model_name)
            response_cost = 0.0
        cost = prompt_cost + response_cost + get_model_request_cost(model_name)
    return cost


class AbstractRouter(ABC):
    @abstractmethod
    def __init__(self, **kwargs) -> None:
        """
        Initialize the routing algorithm, fit the model or load data weights, etc.
        """
        pass

    @abstractmethod
    def batch_route_prompts(self, prompts: list[str], **kwargs) -> NDArray[str]:
        """
        :param prompts: List of prompts to route to.
        :param kwargs: for example, willingness_to_pay
        :return: should return an array of model names (str) in MODELS_TO_ROUTE
        """
        pass
