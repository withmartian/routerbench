import os

import jsonlines
import pandas as pd
import tqdm

from convertors.abstract_convertor import AbstractConvertor
from convertors.common import (calculate_total_cost_for_prompts_and_responses,
                               convert_to_wide_format,
                               generate_oracle_routing_results)


class MartianEvalConvertor(AbstractConvertor):
    def __init__(self, **kwargs):
        super().__init__()

    def convert(self, data_path: str) -> pd.DataFrame:
        data = process_api_logs(data_path)
        input_data = process_raw_data(data)
        models_to_route = input_data.model_name.unique().tolist()
        # Identify IDs with any NaN values in any column
        mask = (
            input_data.isna()
            .groupby(input_data["sample_id"])
            .transform("any")
            .any(axis=1)
        )
        # Filter out these IDs by keeping only rows where the mask is False
        input_data = input_data[~mask]
        input_data = convert_to_wide_format(input_data)
        input_data = calculate_total_cost_for_prompts_and_responses(
            input_data, models_to_route=models_to_route
        )
        input_data = generate_oracle_routing_results(
            input_data, model_to_route=models_to_route
        )
        return input_data


def select_prompts(x):
    # Remove identical prompts in the list
    if isinstance(x, str):
        return [x]
    else:
        # Remove identical prompts in the list, keeping them in order
        tmp_x = []
        for prompt in x:
            if prompt not in tmp_x:
                tmp_x.append(prompt)
        x = tmp_x
    if len(x) == 1:
        return [x[0]]
    elif len(x) == 4:
        # MT Bench
        return [x[0], x[2]]
    else:
        returned_prompts = []
        for prompt in x:
            if "[BEGIN DATA]" in prompt:
                continue
            elif "<|The Start of Assistant " in prompt:
                continue
            elif "\n************\n[END DATA]" in prompt:
                continue
            else:
                returned_prompts.append(prompt)
        return returned_prompts


def extract_prompt_and_response(x):
    if isinstance(x[0], dict):
        prompt = [y["content"] for y in x if y["content"]]
    else:
        prompt = "".join(x)
        # Remove the extra \n that is added by the completion prompt
        prompt = [prompt[:-1]]
    return prompt


def process_api_logs(file_path: str) -> pd.DataFrame:
    file_list = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    # filter the list
    file_list = [path for path in file_list if ".jsonl" in path and "_bak" not in path]

    total_data_list = []
    for file in tqdm.tqdm(file_list):
        list_of_dicts = []
        try:
            with jsonlines.open(file) as reader:
                for record in reader:
                    list_of_dicts.append(record)
        except:
            continue
        if len(list_of_dicts) <= 2:
            continue
        metadata_dict = list_of_dicts[0]["spec"]
        model_name = metadata_dict["completion_fns"][0].replace("raw_models/", "")
        eval_name = metadata_dict["eval_name"]

        eval_set_df = pd.DataFrame(list_of_dicts)
        eval_set_df = eval_set_df[eval_set_df.run_id.notnull()]
        # Iterate over each group of sample_id
        for sample_id, group in eval_set_df.groupby("sample_id"):
            # Initialize an empty dictionary to store the row values
            row_dict = {"sample_id": sample_id}

            # Take the earliest created_at date among the group
            row_dict["created_at"] = group["created_at"].iloc[0]
            row_dict["model_name"] = model_name
            row_dict["eval_name"] = eval_name
            # Have to add to the prompts if multi-turn, it will show up twice
            row_dict["prompt"] = []
            row_dict["sampled"] = []

            # Iterate over each row in the group
            for idx, row in group.iterrows():
                data_dict = row["data"]  # Convert string to dictionary

                # Extract information based on the type
                if row["type"] == "match":
                    row_dict["correct"] = data_dict.get("correct")
                elif row["type"] == "sampling":
                    row_dict["prompt"] += data_dict.get("prompt", [])
                    row_dict["sampled"] += data_dict.get("sampled", [])
                elif row["type"] == "metrics":
                    if "accuracy" in data_dict:
                        row_dict["correct"] = data_dict.get("accuracy")
                    elif "score" in data_dict:
                        # Get the score from the score in the output
                        row_dict["correct"] = data_dict.get("score")
                        row_dict["choice"] = data_dict.get("choice")
                    elif "sacrebleu_sentence_score" in data_dict:
                        row_dict["correct"] = (
                            data_dict.get("sacrebleu_sentence_score") / 100.0
                        )
                    else:
                        raise ValueError(
                            f"Metrics type not recognized {row=}: {data_dict=}"
                        )

            if row_dict.get("correct") is None:
                # Try to deduce the score from the choice, if it is a choice of ABCDE
                # The answer should be 1.0 is A, 0.0 is E
                if (
                    row_dict.get("choice") is not None
                    and row_dict["choice"] != "__invalid__"
                ):
                    row_dict["correct"] = (
                        1.0 - (ord(row_dict["choice"]) - ord("A")) / 4.0
                    )
            # Remove the choice from the row_dict
            row_dict.pop("choice", None)
            # Append the new row to the new DataFrame
            total_data_list.append(row_dict)
    d = pd.DataFrame(total_data_list)
    # d = d[d["sampled"].apply(type) == str]
    d["prompt"] = d.prompt.apply(lambda x: extract_prompt_and_response(x))
    # IF there are multiple prompts, then take the first and third ones
    # d["prompt"] = d.prompt.apply(select_prompts)
    d["prompt"] = d.prompt.apply(lambda x: str(x))
    # If there are multiple responses, then take all but the last one
    d["sampled"] = d.sampled.apply(lambda x: x[:-1] if len(x) > 1 else x)
    d["sampled"] = d.sampled.apply(lambda x: str(x))
    # Remove any rows whose model_name is zero-one-ai/Yi-34B-Chat
    return d


def process_raw_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe[dataframe["correct"].notna()]
    dataframe = dataframe.drop_duplicates(
        subset=["sample_id", "model_name", "prompt"]
    ).reset_index()
    # remove column model_name
    dataframe.index.rename("idx", inplace=True)
    dataframe.drop(columns=["created_at"], inplace=True)
    # Rename sampled to model_response
    dataframe.rename(
        columns={"sampled": "model_response", "correct": "performance"}, inplace=True
    )
    return dataframe
