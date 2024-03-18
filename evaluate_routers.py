import concurrent
import concurrent.futures
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import dotenv
import modal
import numpy as np
import pandas as pd
import tqdm
from jsonargparse import ActionConfigFile, ArgumentParser
from tqdm.auto import tqdm as tqdm_auto

from evaluate_utils import (combined_eval_results_to_eval_collection,
                            generate_knn_routers, generate_mlp_routers,
                            generate_svm_routers, parse_params_from_model_name,
                            save_results, update_df_to_gcs, update_file_to_gcs)

tqdm_auto.pandas()

from embedding.cache import EmbeddingCache
from routers.run_cascading_router import \
    run_cascading_router_for_eval_dataframe
from utils import (WILLINGNESS_TO_PAY, build_train_eval_dataset,
                   get_models_to_route)

warnings.simplefilter("ignore")
dotenv.load_dotenv()

EMBEDDING_CONNECTION_STRING = os.environ["CONNECTION_STRING"]


def run_modal_results(model_params, dataset_df, willingness_to_pay):
    print(f"Evaluating {model_params[-1]} Willingness_to_pay: {willingness_to_pay}")
    model_router_name = model_params[-1]
    modal_factory = modal.Cls.lookup("router_benchmark_paper", "Model")
    model_router = modal_factory(
        router=model_params[0]["router"],  # router
        router_kwargs=model_params[0]["router_kwargs"],  # router_kwargs
        cache_uri=model_params[0]["cache_uri"],  # cache_url
        eval_names=model_params[1],  # train_eval_names
        gcs_path=model_params[0]["gcs_path"],  # gcs_path
        gcp_embedding_paths=model_params[0][
            "gcp_embedding_paths"
        ],  # gcp_embedding_paths
    )
    pay_results = {}

    pay_router_name = model_router_name + f"|willingness_to_pay:{willingness_to_pay}"

    pay_results[pay_router_name] = []
    # Get all available eval names in the dataset
    eval_names = dataset_df.eval_name.unique().tolist()
    results_df = model_router.return_eval_routing.remote(
        eval_names, willingness_to_pay=willingness_to_pay
    )
    # Rename router to router model name
    results_df.rename(columns={"router": pay_router_name}, inplace=True)
    # Check all the sample_ids are in the results_df, with no missing or extra
    assert set(dataset_df["sample_id"].values) == set(results_df["sample_id"].values)
    # Reorder the results_df to match the order of the dataset_df
    results_df = (
        results_df.set_index("sample_id").reindex(dataset_df["sample_id"]).reset_index()
    )

    # Make the result a single list, out of list of subsets
    pay_results[pay_router_name] = results_df[pay_router_name].values.tolist()
    return pay_results


def run_local_results(model_router, model_router_name, dataset_df):
    print(f"Evaluating {model_router_name}")
    pay_results = {}
    # Go through multiple willingness_to_pay values
    for willingness_to_pay in tqdm.tqdm(WILLINGNESS_TO_PAY):
        print(f"Using willingness_to_pay value: {willingness_to_pay}")
        pay_router_name = (
            model_router_name + f"|willingness_to_pay:{willingness_to_pay}"
        )
        pay_results[pay_router_name] = model_router.batch_route_prompts(
            dataset_df["prompt"].values,
            willingness_to_pay=willingness_to_pay,
        )

    return pay_results


def get_results_for_all_evals(
    dataset_df: pd.DataFrame,
    model_routers_and_names: Optional[list[Tuple]] = None,
    use_local: bool = False,
) -> pd.DataFrame:
    """
    Run evaluation and build the evaluation dataframe for all evaluations for all models and routers

    :param dataset_df: Pandas dataframe containing 'embedding', 'sample_id', and model name columns with
    'True' indicating the model got the test correct
    :param model_routers_and_names: List of tuples, containing the model routers\
     (with the batch_route method), and their names
    :param use_willingness_to_pay: Whether to use willingness_to_pay or not
    :return: The original dataframe with additional columns 'oracle', 'oracle|total_cost', 'oracle_accuracy',
    and for each model in model_routers_and_names, 'model_name', 'model_name|total_cost', 'model_name_accuracy'
    """
    # Convert to Wide format, with each model name as a column, do the evaluation, and then convert back to long format
    # Have to make sure it is a copy, not a reference, otherwise it will modify the original dataframe, and lose
    # model responses
    # Other option would be to save the model responses in a separate dataframe, and then join them back together
    # Or put the model_response in a |model_response column

    required_columns = [
        "sample_id",
    ] + MODELS_TO_ROUTE
    for column in required_columns:
        assert (
            column in dataset_df.columns
        ), f"{column} column not found in input dataframe"

    model_router_list = []
    if use_local:
        max_workers = 1
        results_function = run_local_results
        results_inputs = (
            (model_router, model_router_name, dataset_df)
            for model_router, model_router_name in model_routers_and_names
        )
    else:
        max_workers = 1
        results_function = run_modal_results
        results_inputs = (
            (model_params, dataset_df, wp)
            for model_params in model_routers_and_names
            for wp in WILLINGNESS_TO_PAY
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the function to the items
        results = list(
            executor.map(lambda inputs: results_function(*inputs), results_inputs)
        )
        print(f"Finished running {len(results)} results")
    # Go through results, adding into the dataset_df
    for result in results:
        for pay_router_name in result.keys():
            dataset_df[pay_router_name] = result[pay_router_name]
            dataset_df[pay_router_name] = dataset_df[pay_router_name].apply(
                lambda name: name if name in MODELS_TO_ROUTE else None
            )
            dataset_df[pay_router_name + "|total_cost"] = dataset_df.apply(
                lambda row: row[row[pay_router_name] + "|total_cost"],
                axis=1,
            )
            model_router_list.append(pay_router_name)
    dataset_df.rename(columns={"oracle_model_to_route_to": "oracle"}, inplace=True)
    dataset_df["oracle|total_cost"] = dataset_df.apply(
        lambda row: row[row["oracle"] + "|total_cost"]
        if row["oracle"] != "no_model_correct"
        else 0.0,
        axis=1,
    )

    # Drop embedding column if there is one
    if "embedding" in dataset_df.columns:
        dataset_df = dataset_df.drop(columns=["embedding"])

    # replace routing_model column with actual True/False value from the LLM it routes to
    for model_router in model_router_list + ["oracle"]:
        row_indices = np.arange(dataset_df.shape[0])
        col_indices = dataset_df.columns.get_indexer(dataset_df[model_router])
        dataset_df[model_router] = dataset_df.to_numpy()[row_indices, col_indices]
    # transform into long format
    extra_model_names = ["oracle"] + model_router_list
    extra_model_cost_names = ["oracle|total_cost"] + [
        router_name + "|total_cost" for router_name in model_router_list
    ]
    dataset_df = dataset_df.drop_duplicates(subset=["sample_id"])
    llm_total_cost_names = [name + "|total_cost" for name in MODELS_TO_ROUTE]
    eval_names = dataset_df[["sample_id", "eval_name"]]
    df2 = dataset_df.melt(
        id_vars=["sample_id"],
        value_vars=MODELS_TO_ROUTE
        + llm_total_cost_names
        + extra_model_names
        + extra_model_cost_names,
        var_name="model_name",
        value_name="value",
    )
    df2["variable_name"] = df2["model_name"].apply(
        lambda name: "total_cost" if "total_cost" in name else "is_correct"
    )
    df2["model_name"] = df2["model_name"].apply(
        lambda name: name
        if "total_cost" not in name
        else name.replace("|total_cost", "")
    )

    # Calcualte result for each LLM/router, and for each eval set
    df3 = df2.pivot(
        index=["sample_id", "model_name"], columns="variable_name", values="value"
    ).reset_index()
    df3 = df3.merge(eval_names, on="sample_id", how="left")
    # Convert and True/False to 1.0/0.0
    df3["is_correct"] = df3["is_correct"].astype(float)
    aggregated_df = (
        df3.groupby(["eval_name", "model_name"])
        .agg({"is_correct": "mean", "total_cost": "sum"})
        .reset_index()
    )

    # parse routing model parameters
    param_names = []
    for model_router_name in model_router_list:
        param_names += list(parse_params_from_model_name(model_router_name).keys())

    for param_name in param_names:
        aggregated_df[param_name] = pd.NA

    for i in range(aggregated_df.shape[0]):
        model_name = aggregated_df.loc[i, "model_name"]
        params_dict = parse_params_from_model_name(model_name)
        for param_name in params_dict.keys():
            aggregated_df.loc[i, param_name] = params_dict[param_name]

    # replace model_name with the actual model name
    for router_name in model_router_list:
        aggregated_df.loc[
            aggregated_df["model_name"] == router_name, "model_name"
        ] = router_name.split("|")[0]

    aggregated_df = aggregated_df.rename(columns={"is_correct": "performance"})

    return aggregated_df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
    )
    parser.add_argument(
        "--cache-url",
        type=str,
        default=EMBEDDING_CONNECTION_STRING,
        help="The url of the mongodb cache",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Use different fractions of the training set",
    )
    parser.add_argument(
        "--willingness-to-pay",
        action="store_true",
        help="Use different willingness to pay values",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally vs in Modal, if not given, by default runs on modal",
    )
    parser.add_argument(
        "--wanted-eval-name",
        default=None,
        type=Union[str, list[str]],
        help="Run a specific eval, rather than full MARSbench",
    )
    parser.add_argument(
        "--out-of-distribution",
        action="store_true",
        help="Run out-of-distribution experiment",
    )
    parser.add_argument(
        "--gcp-token",
        type=str,
        default="gcs_credentials.json",
        help="The token for the GCP bucket, required is using modal",
    )
    parser.add_argument(
        "--gcp-bucket",
        type=str,
        default="model-saving-bucket",
        help="The bucket to save the data to, for Modal",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default=None,
        help="The name to use when saving the dataframe",
    )
    parser.add_argument(
        "--embedding-models",
        type=Union[list[str], str],
        help="The embedding model(s) to use",
    )
    parser.add_argument("--knn.neighbors", type=list[int])
    parser.add_argument("--knn.metrics", type=list[str])
    parser.add_argument(
        "--mlp.hidden_layer_sizes", type=list[Union[int, Tuple[int, int]]]
    )
    parser.add_argument("--mlp.activation", type=list[str])
    parser.add_argument("--mlp.learning_rates", type=list[float])
    parser.add_argument("--cascading_router.error_rates", type=list[float])
    parser.add_argument(
        "--cascading_router.max_cost_per_response_list", type=list[float]
    )
    parser.add_argument(
        "--local-cache", action="store_true", help="Use purely local cache, not MongoDB"
    )
    parser.add_argument("--config", action=ActionConfigFile)
    args = parser.parse_args()

    # Do setup of directories for later
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists("data/eval_results"):
        os.makedirs("data/eval_results")
    if not os.path.exists(f"data/{args.data_name}/eval_results"):
        os.makedirs(f"data/{args.data_name}/eval_results")
    if not os.path.exists("data/analysis_results/"):
        os.makedirs("data/analysis_results")

    # save the config to the data folder
    parser.save(
        args,
        f"data/{args.data_name}/evaluate_routers.yaml",
        format="yaml",
        overwrite=True,
    )
    # Set the parameters here (for knn)
    neighbors = args.knn.neighbors
    metrics = args.knn.metrics
    hidden_layer_sizes = args.mlp.hidden_layer_sizes
    activations = args.mlp.activation
    learning_rates = args.mlp.learning_rates

    embedding_models = (
        args.embedding_models if args.embedding_models else ["all-MiniLM-L6-v2"]
    )

    cache = EmbeddingCache(args.cache_url, local_mode=args.local_cache)

    if args.data_path.endswith(".csv"):
        dataset_df = pd.read_csv(args.data_path)
    else:
        dataset_df = pd.read_pickle(args.data_path)

    MODELS_TO_ROUTE = get_models_to_route(dataset_df)
    train_test_split = True
    dataset_df_train = dataset_df
    dataset_df_eval = dataset_df

    import time

    output_filenames = []
    t = time.time()
    if args.wanted_eval_name is not None:
        if isinstance(args.wanted_eval_name, list):
            wanted_eval_names = args.wanted_eval_name
        else:
            wanted_eval_names = [args.wanted_eval_name]
    else:
        wanted_eval_names = dataset_df.eval_name.unique().tolist()
    if args.out_of_distribution:
        other_eval_names = dataset_df.eval_name.unique().tolist()
    else:
        other_eval_names = wanted_eval_names
    fraction = args.train_fraction

    # Upload to GCP the embedding files (if they exist) and the converted file
    # Modal parameterization includes the path to the GCP files
    # On startup, it downloads those files and does the whole generation
    # IF running locally, ignores all that and continues on without it
    if not args.local:
        # Upload the converted file to GCP
        # Get the path without the file extension
        gcs_path = update_df_to_gcs(
            dataset_df,
            str(Path(args.data_path).stem),
            args.data_path.split("/")[-1],
            token=args.gcp_token,
            bucket=args.gcp_bucket,
        )
        # Upload the embedding files to GCP as well
        embedding_paths = []
        for embedding_model in embedding_models:
            if not os.path.exists(f"data/embedding_cache_{embedding_model}.pkl"):
                continue
            embedding_paths.append(
                update_file_to_gcs(
                    f"data/embedding_cache_{embedding_model}.pkl",
                    gcs_dir=os.path.join(
                        args.data_path.split("/")[-1], str(Path(args.data_path).stem)
                    ),
                    filename=f"embedding_cache_{embedding_model}.pkl",
                    bucket=args.gcp_bucket,
                    token=args.gcp_token,
                )
            )

    for wanted_eval_name in wanted_eval_names:
        model_routers_and_names = []
        all_parameter_combinations = []
        dataset_df_train, dataset_df_eval = build_train_eval_dataset(
            wanted_eval_name, other_eval_names, dataset_df, fraction=fraction
        )

        if args.local:
            knn_param_combinations = [
                (e_model, n, m)
                for e_model in embedding_models
                for n in neighbors
                for m in metrics
            ]
            mlp_param_combinations = [
                (e_model, n, m, lr)
                for e_model in embedding_models
                for n in hidden_layer_sizes
                for m in activations
                for lr in learning_rates
            ]
            svm_param_combinations = embedding_models
            model_routers_and_names += generate_svm_routers(
                svm_param_combinations,
                cache,
                dataset_df_train,
                fraction,
                MODELS_TO_ROUTE,
            )
            model_routers_and_names += generate_mlp_routers(
                mlp_param_combinations,
                cache,
                dataset_df_train,
                fraction,
                MODELS_TO_ROUTE,
            )
            model_routers_and_names += generate_knn_routers(
                knn_param_combinations,
                cache,
                dataset_df_train,
                fraction,
                MODELS_TO_ROUTE,
            )
        else:
            knn_param_combinations = [
                (
                    {
                        "router": "knn",
                        "router_kwargs": {
                            "embedding_model": e_model,
                            "n_neighbors": n,
                            "distance_metric": m,
                            "fraction": fraction,
                            "models_to_route": MODELS_TO_ROUTE,
                            "out_of_distribution": args.out_of_distribution,
                        },
                        "cache_uri": EMBEDDING_CONNECTION_STRING,
                        "gcs_path": gcs_path,
                        "gcp_embedding_paths": embedding_paths,
                    },
                    [wanted_eval_name],
                    f"knn|neighbors:{n}|metric:{m}|embedding:{e_model}|fraction:{fraction}",
                )
                for e_model in embedding_models
                for n in neighbors
                for m in metrics
            ]
            mlp_param_combinations = [
                (
                    {
                        "router": "mlp",
                        "router_kwargs": {
                            "embedding_model": e_model,
                            "hidden_layer_sizes": n,
                            "activation": m,
                            "learning_rate": lr,
                            "fraction": fraction,
                            "models_to_route": MODELS_TO_ROUTE,
                            "out_of_distribution": args.out_of_distribution,
                        },
                        "cache_uri": EMBEDDING_CONNECTION_STRING,
                        "gcs_path": gcs_path,
                        "gcp_embedding_paths": embedding_paths,
                    },
                    [wanted_eval_name],
                    f"mlp|hidden_layer_sizes:({n})|activation_function:{m}|learning_rate_method:constant|learning_rate:{lr}|embedding:{e_model}|fraction:{fraction}",
                )
                for e_model in embedding_models
                for n in hidden_layer_sizes
                for m in activations
                for lr in learning_rates
            ]
            svm_param_combinations = [
                (
                    {
                        "router": "svm",
                        "router_kwargs": {
                            "embedding_model": e_model,
                            "fraction": fraction,
                            "models_to_route": MODELS_TO_ROUTE,
                            "out_of_distribution": args.out_of_distribution,
                        },
                        "cache_uri": EMBEDDING_CONNECTION_STRING,
                        "gcs_path": gcs_path,
                        "gcp_embedding_paths": embedding_paths,
                    },
                    [wanted_eval_name],
                    f"svm|embedding:{e_model}|fraction:{fraction}",
                )
                for e_model in embedding_models
            ]
            all_parameter_combinations += knn_param_combinations
            all_parameter_combinations += mlp_param_combinations
            all_parameter_combinations += svm_param_combinations

        print(f"time to train models: {time.time() - t}")
        print(f"Number of combinations: {len(all_parameter_combinations)}")
        print("evaluating models")
        result_df = get_results_for_all_evals(
            dataset_df_eval,
            model_routers_and_names=all_parameter_combinations
            if all_parameter_combinations
            else model_routers_and_names,
            use_local=args.local,
        )
        output_filenames.append(
            save_results(
                result_df,
                train_test_split,
                eval_name=wanted_eval_name,
            )
        )
        output_filenames.append(
            run_cascading_router_for_eval_dataframe(
                dataset_df_eval,
                eval_name=wanted_eval_name,
                max_cost_per_response_list=args.cascading_router.max_cost_per_response_list,
                error_rates=args.cascading_router.error_rates,
                models_to_route=MODELS_TO_ROUTE,
            )
        )
    # Combine output results into a single file
    combined_eval_results_to_eval_collection(
        output_filenames,
        models_to_route=MODELS_TO_ROUTE,
        data_name=args.data_name if args.data_name else args.wanted_eval_names,
    )
