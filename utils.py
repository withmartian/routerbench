import os
from datetime import datetime

import numpy as np
import pandas as pd

WILLINGNESS_TO_PAY = [
    0.0000000001,
    0.000000001,
    0.00000001,
    0.0000001,
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.005,
    0.01,
    0.1,
    0.5,
    1.0,
    1.5,
    2.0,
    10.0,
    50.0,
    100.0,
    500.0,
    1000.0,
    2000.0,
    3000.0,
    4000.0,
    5000.0,
    7500.0,
    10000.0,
    25000.0,
    50000.0,
    75000.0,
    100000.0,
    1000000.0,
    1000000000.0,
]


def parse_model_parameters(model_string: str) -> dict:
    """
    example input:
    model_string = 'knn|neighbors:500.0|metric:cosine|fraction:1.0|embedding:all-MiniLM-L6-v2|willingness_to_pay:2.0|total_cost'

    output:
    {'routing_algorithm': 'knn',
     'neighbors': '500.0',
     'metric': 'cosine',
     'fraction': '1.0',
     'embedding': 'all-MiniLM-L6-v2',
     'willingness_to_pay': '2.0',
     'value_type': 'total_cost'}
    """
    parts = model_string.split("|")
    assert len(parts) >= 2, "Model string must have at least 2 parts"
    assert (
        isinstance(parts[0], str) == 1 and isinstance(parts[-1], str) == 1
    ), "The first and last parts must have 1 element only"

    output_dict = {}
    output_dict["routing_algorithm"] = parts[0]
    output_dict["value_type"] = parts[-1]
    model_parameters = [parameter.split(":") for parameter in parts[1:-1]]
    model_parameters = {key: value for key, value in model_parameters}
    return {**output_dict, **model_parameters}


def save_data_to_file(
    data,
    save_path: str,
    base_name: str,
    data_name: str,
    format="csv",
    hour=True,
    minute=False,
):
    """
    Save the dataframe as a csv file, and print the shape of the dataframe.

    :param df: Dataframe or data object to save
    :param save_path: Path to save the dataframe
    :param base_name: Base name of the csv file
    :return:
    """
    from evaluation.eval import EvaluationResultCollection

    file_name_raw = f"{base_name}__{generate_datetime_str(hour, minute)}__{data_name}"
    if isinstance(data, pd.DataFrame):
        if format == "csv":
            file_name = f"{file_name_raw}.csv"
            data.to_csv(os.path.join(save_path, file_name), index=False)
        elif format == "pkl":
            file_name = f"{file_name_raw}.pkl"
            data.to_pickle(os.path.join(save_path, file_name))
    elif isinstance(data, EvaluationResultCollection):
        assert base_name == "eval_results"
        if format == "pkl":
            file_name = f"{file_name_raw}.pkl"
            data.to_pickle(os.path.join(save_path, file_name))
        elif format == "csv":
            file_name = f"{file_name_raw}.csv"
            data.to_csv(os.path.join(save_path, file_name))

    print(f"Saved to: {os.path.join(save_path, file_name)}")


def generate_datetime_str(hour=True, minute=False) -> str:
    """
    Generate a random datetime string in the format of %b%d:%H:%M

    """
    today_date = datetime.utcnow()
    if hour:
        if minute:
            date_string = today_date.strftime("%m-%d-%H-%M")
        else:
            date_string = today_date.strftime("%m-%d-%H")
    else:
        date_string = today_date.strftime("%m-%d")
    return date_string


def get_models_to_route(dataset_df: pd.DataFrame) -> list[str]:
    """
    Get the models to route from the dataset_df

    :param dataset_df: The dataset_df
    :return: The models to route
    """
    # Model names are ones that are in two columns, that end with |model_response and |total_cost
    return [
        col.replace("|model_response", "")
        for col in dataset_df.columns
        if "|model_response" in col
    ]


def build_train_eval_dataset(
    wanted_eval_name,
    other_eval_names,
    dataset_df,
    fraction=0.7,
    out_of_distribution=False,
):
    all_train_sample_ids = []
    for other_eval_name in other_eval_names:
        if out_of_distribution:
            if (
                other_eval_name == wanted_eval_name
            ):  # Don't train on samples from the eval that is being tested
                continue
        else:
            if (
                other_eval_name != wanted_eval_name
            ):  # Only train on samples from the eval that is being tested
                continue
        eval_names = (
            dataset_df["eval_name"][
                dataset_df.eval_name.str.startswith(other_eval_name).tolist()
            ]
            .unique()
            .tolist()
        )
        # Train eval names is all eval names in l other than ones that start with fraction
        # Go per eval name, get the percentage of sample_ids that are in that eval name
        eval_sample_ids = (
            dataset_df[dataset_df.eval_name.isin(eval_names)]
            .sample_id.unique()
            .tolist()
        )
        # Get the unique number of sources
        if "source" in dataset_df.columns:
            eval_sources = (
                dataset_df[dataset_df.eval_name.isin(eval_names)]
                .source.unique()
                .tolist()
            )
            print(f"Unique sources in {other_eval_name}: {len(eval_sources)}")
            np.random.seed(42)
            np.random.shuffle(eval_sources)
            train_eval_sources = eval_sources[: int(len(eval_sources) * fraction)]
            all_train_sample_ids += (
                dataset_df[dataset_df.source.isin(train_eval_sources)]
                .sample_id.unique()
                .tolist()
            )
            print(f"Total Training Tasks: {len(all_train_sample_ids)}")
        else:
            # Set random seed
            np.random.seed(42)
            np.random.shuffle(eval_sample_ids)
            all_train_sample_ids += eval_sample_ids[
                : int(len(eval_sample_ids) * fraction)
            ]
        # all_test_sample_ids += eval_sample_ids[int(len(eval_sample_ids) * fraction):]
    eval_names = (
        dataset_df["eval_name"][
            dataset_df.eval_name.str.startswith(wanted_eval_name).tolist()
        ]
        .unique()
        .tolist()
    )
    # Train eval names is all eval names in l other than ones that start with fraction
    # Go per eval name, get the percentage of sample_ids that are in that eval name
    # all_train_sample_ids = []
    all_test_sample_ids = []
    if "source" in dataset_df.columns:
        eval_sources = (
            dataset_df[dataset_df.eval_name.isin(eval_names)].source.unique().tolist()
        )
        print(f"Unique sources in {other_eval_name}: {len(eval_sources)}")
        np.random.seed(42)
        np.random.shuffle(eval_sources)
        test_eval_sources = eval_sources[int(len(eval_sources) * fraction) :]
        all_test_sample_ids += (
            dataset_df[dataset_df.source.isin(test_eval_sources)]
            .sample_id.unique()
            .tolist()
        )
    else:
        eval_sample_ids = (
            dataset_df[dataset_df.eval_name.isin(eval_names)]
            .sample_id.unique()
            .tolist()
        )
        np.random.seed(42)
        np.random.shuffle(eval_sample_ids)
        # all_train_sample_ids += eval_sample_ids[: int(len(eval_sample_ids) * fraction)]
        all_test_sample_ids += eval_sample_ids[int(len(eval_sample_ids) * fraction) :]
    train_eval_names = all_train_sample_ids
    validation_eval_names = all_test_sample_ids
    validation_set = set(validation_eval_names)
    train_eval_names = set(train_eval_names)
    print(f"Total Training Tasks: {len(train_eval_names)}")
    dataset_df_eval = dataset_df[
        dataset_df["sample_id"].apply(lambda name: name in validation_set)
    ]
    dataset_df_train = dataset_df[
        dataset_df["sample_id"].apply(lambda name: name in train_eval_names)
    ]

    # Set the eval_name to the wanted_eval_name
    dataset_df_eval.eval_name = wanted_eval_name
    dataset_df_train.eval_name = wanted_eval_name

    print(dataset_df_eval.eval_name.unique())
    print(dataset_df_train.eval_name.unique())
    print(f"Total Training Samples: {len(dataset_df_train)}")
    print(f"Total Validation Samples: {len(dataset_df_eval)}")
    # Check that there are no overlapping sample_ids in the train and test dataset
    assert (
        len(
            set(dataset_df_train.sample_id.unique()).intersection(
                set(dataset_df_eval.sample_id.unique())
            )
        )
        == 0
    )
    return dataset_df_train, dataset_df_eval
