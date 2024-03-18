"""
This file does the following
1. plot the accuracy difference in accuracy (and cost) between different models more easily
2. This is good for comparing the accuracy of different models, but no good to explore deeper on the specific metrics
"""
import os.path
from typing import Union

import pandas as pd
from jsonargparse import ActionConfigFile, ArgumentParser

from evaluation.eval import EvaluationResultCollection

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        help="input file path to saved EvaluationCollection",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="input file path to the original input data file, that has a `model_name` column with the model names",
    )
    parser.add_argument(
        "--wanted-eval-names",
        type=Union[str, list[str]],
        help="evaluation name to plot, if None, plots all evals",
        default=None,
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="default",
        help="The name to use when saving in a subfolder",
    )
    parser.add_argument(
        "--config", action=ActionConfigFile, help="path to the configuration file"
    )
    args = parser.parse_args()
    save_path = f"data/{args.data_name}/analysis_results/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        # save the config to the data folder
    parser.save(
        args,
        f"data/{args.data_name}/visualize_results.yaml",
        format="yaml",
        overwrite=True,
    )
    # Load the EvaluationResultCollection
    evaluation_collection = EvaluationResultCollection.load(args.input_file)
    # Get all the router names in the collection
    router_names = evaluation_collection.get_router_names()
    if args.data_path.endswith(".csv"):
        dataset_df = pd.read_csv(args.data_path)
    else:
        dataset_df = pd.read_pickle(args.data_path)
    if args.wanted_eval_names is None:
        eval_names_to_plot = evaluation_collection.get_eval_names()
    else:
        if isinstance(args.wanted_eval_names, str):
            eval_names_to_plot = [args.wanted_eval_names]
        else:
            eval_names_to_plot = args.wanted_eval_names
    for name in eval_names_to_plot:
        evaluation_collection.plot_performance_vs_cost(
            save_file_path=save_path,
            show_plot=True,
            eval_name=name,
            plot_ndch=True,
            models_to_route=dataset_df.model_name.unique().tolist(),
        )
