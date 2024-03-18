from datetime import datetime

import pandas as pd

model_metric_dict = {
    "knn": ["neighbors", "metric", "embedding", "willingness_to_pay"],
    "mlp": [
        "hidden_layer_sizes",
        "activation_function",
        "learning_rate_method",
        "learning_rate",
        "embedding",
        "willingness_to_pay",
    ],
    "cascading router": [
        "max_cost_per_response",
        "evaluator_error_rate",
        "performance_threshold",
    ],
    "svm": ["embedding", "willingness_to_pay"],
}


def transform_single_model_df(original_long_df, router_model_name, model_metric_dict):
    model_df = original_long_df[
        original_long_df["model_name"] == router_model_name
    ].copy()
    model_df["variable"] = router_model_name
    for param in model_metric_dict[router_model_name]:
        model_df["variable"] = (
            model_df["variable"] + "|" + param + ":" + model_df[param].astype(str)
        )
    all_router_model_params_name = [
        item for sublist in model_metric_dict.values() for item in sublist
    ]
    for param in all_router_model_params_name:
        if param in model_df.columns:
            model_df.drop(param, axis=1, inplace=True)
    return model_df


def transform_batch_df_to_wide_format(raw_long_df, model_metric_dict):
    router_model_dfs = []
    for router_model_name in model_metric_dict.keys():
        model_df = transform_single_model_df(
            raw_long_df, router_model_name, model_metric_dict
        )
        router_model_dfs.append(model_df)
    router_model_df = pd.concat(router_model_dfs)
    router_model_df.drop("model_name", axis=1, inplace=True)
    router_model_df.rename(columns={"variable": "model_name"}, inplace=True)
    router_model_df_long = router_model_df.melt(
        id_vars=["model_name", "eval_name"],
        value_vars=["performance", "total_cost"],
        var_name="measure",
        value_name="value",
    )

    # Now combine with the rows where model_name is not in model_metric_dict (these don't need the transformation)
    raw_models_df = raw_long_df[
        ~raw_long_df["model_name"].isin(model_metric_dict.keys())
    ].melt(
        id_vars=["eval_name", "model_name"],
        value_vars=["performance", "total_cost"],
        var_name="measure",
        value_name="value",
    )

    # Concatenate all the dataframes
    long_df = pd.concat([router_model_df_long, raw_models_df], ignore_index=True)

    long_df["model_name2"] = long_df["model_name"] + "|" + long_df["measure"]

    df3 = long_df.pivot_table(
        index=["eval_name"], columns=["model_name2"], values="value"
    ).reset_index()

    return df3


def get_average_accuracy(df_to_plot):
    """
    Get the average accuracy for each model
    df_to_plot: the dataframe to plot, should be in wide format
    """
    all_columns = list(df_to_plot.columns)
    raw_model_accuracy_cols = [col for col in all_columns if "accuracy" in col]
    df = df_to_plot[["eval_name"] + raw_model_accuracy_cols]
    # Get average value for each column
    # Want the average accuracy for each column, across all eval_name rows (axis=0)
    #
    df["average_accuracy"] = df[raw_model_accuracy_cols].mean(axis=0)
    return df[["eval_name", "average_accuracy"]]


def generate_datetime_str():
    today_date = datetime.now()
    date_string = today_date.strftime("%b%d")
    return date_string
