import os
from datetime import datetime

import gcsfs
import pandas as pd
import tqdm

from evaluation.eval import DefaultEvaluationResult, EvaluationResultCollection
from routers.knn_router import KNNRouter
from routers.mlp_router import MLPRouter
from routers.svm_router import SVMRouter
from utils import save_data_to_file


def save_results(result_df, train_test_split=False, eval_name: str = ""):
    today_date = datetime.now()
    date_string = today_date.strftime("%b%d:%H:%M")
    output_filename = f"data/eval_results/eval_results-eval-{eval_name}-{date_string}{'-val_split' if train_test_split else ''}.csv"
    result_df.to_csv(
        output_filename,
        index=False,
    )

    print(
        "Saved to: ",
        output_filename,
    )
    return output_filename


def combined_eval_results_to_eval_collection(
    output_filenames,
    data_name: str,
    models_to_route: list[str] = (" ",),
):
    llm_names = models_to_route + ["oracle"]
    dfs = []
    for f in output_filenames:
        df = pd.read_csv(f)
        # Rename accuracy to performance, if accuracy is in list
        if "accuracy" in df.columns:
            df = df.rename(columns={"accuracy": "performance"})
        dfs.append(df)
    result_df = pd.concat(dfs)
    # Drop duplicate rows
    result_df = result_df.drop_duplicates()
    result_df = result_df.sort_values(by=["eval_name"])

    # Select all rows where model name is in llm_names
    llm_results = result_df[result_df["model_name"].isin(llm_names)]
    evaluation_results = []

    # Get all model names that are not in llm_names
    model_names = [
        name for name in result_df.model_name.unique() if name not in llm_names
    ]

    for model_name in model_names:
        single_result_df = result_df[result_df["model_name"] == model_name]
        # Combine with llm results
        single_result_df = pd.concat([single_result_df, llm_results])
        evaluation_result = DefaultEvaluationResult(
            router_name=model_name, results=single_result_df, metrics=["accuracy"]
        )
        evaluation_results.append(evaluation_result)
    all_evaluation_results = EvaluationResultCollection(evaluation_results)
    save_data_to_file(
        data=all_evaluation_results,
        save_path=f"data/{data_name}/eval_results/",
        base_name="eval_results",
        data_name=data_name,
        format="csv",
    )
    save_data_to_file(
        data=all_evaluation_results,
        save_path=f"data/{data_name}/eval_results/",
        base_name="eval_results",
        data_name=data_name,
        format="pkl",
    )


def generate_mlp_routers(
    parameter_combinations, cache, train_df, fraction, models_to_route
):
    print("training mlp models")
    model_routers_and_names = []
    for embed_model, hidden_layer_size, activation, learning_rate in tqdm.tqdm(
        parameter_combinations
    ):
        mlp_router = MLPRouter(
            train_file=train_df,
            hidden_layer_sizes=hidden_layer_size,
            activation_function=activation,
            learning_rate=learning_rate,
            embedding_model=embed_model,
            cache=cache,
            models_to_route=models_to_route,
        )
        # The name of the model should follow this format for easy parsing later
        # {model_name}|{param1_name}:{param1_value}|{param2_name}:{param2_value}
        hidden_layer_size_name = f"{hidden_layer_size},"
        mlp_name = f"mlp|hidden_layer_sizes:({hidden_layer_size_name})|activation_function:{activation}|learning_rate_method:constant|learning_rate:{learning_rate}|embedding:{embed_model}|fraction:{fraction}"
        model_routers_and_names.append((mlp_router, mlp_name))
    return model_routers_and_names


def parse_params_from_model_name(model_name: str) -> dict:
    """
    Parse the parameters from the model name

    :param model_name: The name of the model, should follow this format:
    {model_name}|{param1_name}:{param1_value}|{param2_name}:{param2_value}
    :return: A dictionary of the parameters and their values
    """
    if ":" not in model_name:
        return {}
    params = {}
    model_name = model_name.split("|")
    for param in [name for name in model_name if ":" in name]:
        param = param.split(":")
        params[param[0]] = param[1]
    return params


def generate_knn_routers(
    parameter_combinations, cache, train_df, fraction, models_to_route
):
    print("training knn models")
    model_routers_and_names = []
    for embed_model, neighbor_count, metric_name in tqdm.tqdm(parameter_combinations):
        knn = KNNRouter(
            train_file=train_df,
            n_neighbors=neighbor_count,
            distance_metric=metric_name,
            embedding_model=embed_model,
            cache=cache,
            models_to_route=models_to_route,
        )
        # The name of the model should follow this format for easy parsing later
        # {model_name}|{param1_name}:{param1_value}|{param2_name}:{param2_value}
        knn_name = f"knn|neighbors:{neighbor_count}|metric:{metric_name}|embedding:{embed_model}|fraction:{fraction}"
        model_routers_and_names.append((knn, knn_name))
    return model_routers_and_names


def generate_svm_routers(
    parameter_combinations, cache, train_df, fraction, models_to_route
):
    print("training svm models")
    model_routers_and_names = []
    for embed_model in tqdm.tqdm(parameter_combinations):
        svm = SVMRouter(
            train_file=train_df,
            embedding_model=embed_model,
            cache=cache,
            models_to_route=models_to_route,
        )
        # The name of the model should follow this format for easy parsing later
        # {model_name}|{param1_name}:{param1_value}|{param2_name}:{param2_value}
        svm_name = f"svm|embedding:{embed_model}|fraction:{fraction}"
        model_routers_and_names.append((svm, svm_name))
    return model_routers_and_names


def update_df_to_gcs(df, gcs_dir, filename, bucket, token):
    fs = gcsfs.GCSFileSystem(project=bucket, token=token)
    final_gcs_dir = os.path.join(gcs_dir, filename)
    full_gcs_path = f"gcs://{bucket}/" + final_gcs_dir
    # Check if file already exists in GCS
    if fs.exists(full_gcs_path):
        print(f"File already exists in {full_gcs_path}")
        return full_gcs_path
    print(final_gcs_dir)
    with fs.open(full_gcs_path, "wb") as f:
        # Write the DataFrame to GCS
        df.to_pickle(f)
    print(f"Uploaded DataFrame to {full_gcs_path}")
    return full_gcs_path


def update_file_to_gcs(filepath, gcs_dir, filename, bucket, token):
    fs = gcsfs.GCSFileSystem(project=bucket, token=token)
    final_gcs_dir = os.path.join(gcs_dir, filename)
    full_gcs_path = f"gcs://{bucket}/" + final_gcs_dir
    # Check if file already exists in GCS
    if fs.exists(full_gcs_path):
        print(f"File already exists in {full_gcs_path}")
        return full_gcs_path
    print(final_gcs_dir)
    with fs.open(full_gcs_path, "wb") as f:
        # Write the DataFrame to GCS
        with open(filepath, "rb") as local_file:
            f.write(local_file.read())
    print(f"Uploaded DataFrame to {full_gcs_path}")
    return full_gcs_path
