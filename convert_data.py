import os
import warnings
from typing import Union

import dotenv
import pandas as pd
from jsonargparse import ActionConfigFile, ArgumentParser
from tqdm.auto import tqdm as tqdm_auto

from convertors import MartianEvalConvertor, OpenAIEvalsConvertor
from embedding.cache import EmbeddingCache
from utils import get_models_to_route, save_data_to_file

tqdm_auto.pandas()


warnings.simplefilter("ignore")
dotenv.load_dotenv()

EMBEDDING_CONNECTION_STRING = os.environ["CONNECTION_STRING"]


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
        "--input-format",
        type=str,
        default="openai",
    )
    # add a task_name argument
    parser.add_argument(
        "--data-name",
        type=str,
        default="default",
        help="The name to use when saving the dataframe",
    )
    parser.add_argument(
        "--embedding-models",
        type=Union[list[str], str],
        help="The embedding model(s) to use",
    )
    parser.add_argument(
        "--local-cache", action="store_true", help="Use purely local cache, not MongoDB"
    )
    parser.add_argument("--config", action=ActionConfigFile)

    args = parser.parse_args()
    if not os.path.exists(f"data/{args.data_name}"):
        os.makedirs(f"data/{args.data_name}")
    # save the config to the data folder
    parser.save(
        args, f"data/{args.data_name}/convert_data.yaml", format="yaml", overwrite=True
    )

    embedding_models = (
        args.embedding_models if args.embedding_models else ["all-MiniLM-L6-v2"]
    )
    cache = EmbeddingCache(args.cache_url, local_mode=args.local_cache)

    if args.data_path.endswith(".csv"):
        dataset_df = pd.read_csv(args.data_path)
    elif args.data_path.endswith(".pkl"):
        dataset_df = pd.read_pickle(args.data_path)
    else:
        # Dataset is the raw Martian Evals output
        dataset_df = args.data_path

    if args.input_format == "martian-evals":
        convertor = MartianEvalConvertor()
    else:
        convertor = OpenAIEvalsConvertor()

    dataset_df = convertor.convert_and_embed(
        dataset_df, cache, embedding_models=embedding_models
    )

    # Get the models to route
    MODELS_TO_ROUTE = get_models_to_route(dataset_df)

    for model_name in MODELS_TO_ROUTE:
        dataset_df[model_name].fillna(0.0, inplace=True)

    # Save out the converted data
    save_data_to_file(
        dataset_df,
        save_path=f"data/{args.data_name}/",
        base_name="input_wide",
        format="pkl",
        data_name=args.data_name,
    )
    print("Done!")
