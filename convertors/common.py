import numpy as np
import pandas as pd

from routers.abstract_router import calculate_cost_for_prompt_and_response


def calculate_total_cost_for_prompts_and_responses(
    dataset_df: pd.DataFrame, models_to_route: list[str]
) -> pd.DataFrame:
    print("Calculating total cost for prompts and responses...")
    print(f"in total {len(models_to_route)} models")
    for model_name in models_to_route:
        print(f"calculating for {model_name}")
        dataset_df[model_name + "|total_cost"] = dataset_df.progress_apply(
            lambda row: calculate_cost_for_prompt_and_response(
                row[
                    "prompt"
                ],  # A string representation of a list, so convert back to list
                row[model_name + "|model_response"],
                model_name,
            ),
            axis=1,
        )
    return dataset_df


def convert_to_wide_format(
    dataset_df: pd.DataFrame, with_cost_col: bool = False
) -> pd.DataFrame:
    """
    Convert from the long format to the wide format, with each model name as a column, and the model responses recorded

    :param dataset_df: Dataframe in the wide format
    :param with_cost_col: Whether to include the cost column
    :return:
    """
    model_responses = []
    model_specific_columns = (
        ["model_response", "cost"] if with_cost_col else ["model_response"]
    )
    for model_name in dataset_df["model_name"].unique():
        # Select all rows that have the model_name
        columns_to_take = [
            "sample_id",
            "eval_name",
        ] + model_specific_columns
        sample_id_and_model_responses = dataset_df[
            dataset_df["model_name"] == model_name
        ][columns_to_take]
        columns_to_rename = {}
        for col in model_specific_columns:
            columns_to_rename[col] = (
                model_name + "|" + col if col != "cost" else model_name + "|total_cost"
            )
        sample_id_and_model_responses.rename(columns=columns_to_rename, inplace=True)
        model_responses.append(sample_id_and_model_responses)
    # Drop model_response and cost columns
    dataset_df = dataset_df.drop(columns=model_specific_columns)
    df_pivot = dataset_df.pivot_table(
        index=["sample_id", "prompt", "eval_name"],
        columns="model_name",
        values="performance",
        aggfunc=np.mean,
    ).reset_index()
    df_pivot.index.rename("idx", inplace=True)

    # drop column that contain any nan value
    for model_response in model_responses:
        df_pivot = df_pivot.merge(
            model_response,
            on=["sample_id", "eval_name"],
            how="left",
        )
    return df_pivot


def get_highest_accuracy_lowest_cost(row: pd.Series, models_to_route: list[str]) -> str:
    """
    Get the model with the highest accuracy and lowest cost for a given row.

    :param row: Row of the dataset_df dataframe
    :return: The model with the highest accuracy and lowest cost
    """
    # Check accuracy first, get the model(s) with the highest accuracy
    max_accuracy = row[models_to_route].max()
    # Get all the models with max accuracy, among them, choose the cheapest
    return models_to_route[
        np.argmin(
            [
                row[model_name + "|total_cost"]
                if row[model_name] == max_accuracy
                else 1e10
                for model_name in models_to_route
            ]
        )
    ]


def generate_oracle_routing_results(
    dataset_df: pd.DataFrame, model_to_route: list[str]
):
    """
    Generates the oracle routing values for the dataset, and adds them to the dataset_df dataframe.

    This can be used for supervised learning, so the models learn the cheapest, best routing available.


    :param dataset_df: Dataframe containing the 'embedding' and 'sample_id' columns
    :param model_to_route: List of models to route to
    :return: Dataframe containing the model the oracle would route to, or 'no_model_correct' if no model correctly answered the question
    """
    print("Generating oracle routing results...")
    # For each row, get the model that has the highest accuracy, if there are multiple tied ones, get the cheapest one
    dataset_df["oracle_model_to_route_to"] = dataset_df.progress_apply(
        lambda row: get_highest_accuracy_lowest_cost(
            row, models_to_route=model_to_route
        )
        if any(row[model_to_route])
        else "no_model_correct",
        axis=1,
    )
    return dataset_df
