import os

import dotenv
import modal
import pandas as pd
from modal import Image, Secret, method

from embedding.cache import EmbeddingCache
from utils import build_train_eval_dataset, get_models_to_route

dotenv.load_dotenv()

EMBEDDING_CONNECTION_STRING = os.environ["CONNECTION_STRING"]
GCP_BUCKET_NAME = os.environ["GCP_BUCKET_NAME"]
GCS_CREDENTIALS_PATH = "gcs_credentials.json"


def download_model_to_folder(
    gcs_model_path: str, local_path: str, gcs_embedding_paths: list[str] = None
):
    import json

    from google.cloud import storage

    with open(GCS_CREDENTIALS_PATH, "w") as f:
        json.dump(json.loads(os.environ["CREDENTIALS"]), f)

    storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS_PATH)
    bucket = storage_client.get_bucket(GCP_BUCKET_NAME)
    prefixed_gcs_model_path = gcs_model_path.split(f"{GCP_BUCKET_NAME}/")[-1]
    blob = list(bucket.list_blobs(prefix=prefixed_gcs_model_path))[0]
    print(f"downloading {blob.name} to {local_path}")
    blob.download_to_filename(local_path)
    os.mkdir("data")
    if gcs_embedding_paths is not None:
        for gcs_embedding_path in gcs_embedding_paths:
            prefixed_gcs_embedding_path = gcs_embedding_path.split(
                f"{GCP_BUCKET_NAME}/"
            )[-1]
            blob = list(bucket.list_blobs(prefix=prefixed_gcs_embedding_path))[0]
            print(f"downloading {blob.name} to data/{blob.name.split('/')[-1]}")
            blob.download_to_filename(f"data/{blob.name.split('/')[-1]}")


pandas_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "pandas",
        "tokencost",
        "scikit-learn",
        "sentence_transformers",
        "requests",
        "matplotlib",
        "python-dotenv",
        "typing",
        "google-cloud-storage",
        "pymongo",
        "voyageai",
        "tqdm",
        "poetry",
    )
    .copy_mount(
        modal.Mount.from_local_dir("./", remote_path="/root/alt-routing-methods")
    )
    .run_commands("cd /root/alt-routing-methods && pip install -e .")
)

stub = modal.Stub(name="routerbench")


@stub.cls(
    image=pandas_image,
    container_idle_timeout=150,
    timeout=18000,
    cpu=2,
    memory=1024,
    secrets=[Secret.from_name("gcs-credentials")],
)
class Model:
    def __init__(
        self,
        router: str,
        router_kwargs: dict,
        cache_uri: str,
        gcs_path: str,
        gcp_embedding_paths: list[str] = None,
        eval_names: list[str] = None,
    ) -> None:
        os.environ["CONNECTION_STRING"] = EMBEDDING_CONNECTION_STRING
        from routers.knn_router import KNNRouter
        from routers.mlp_router import MLPRouter
        from routers.svm_router import SVMRouter

        out_of_distribution = router_kwargs.get("out_of_distribution", False)
        local_path = gcs_path.split("/")[-1]
        download_model_to_folder(
            gcs_path, local_path=local_path, gcs_embedding_paths=gcp_embedding_paths
        )
        self.dataset_df = pd.read_pickle(local_path)
        self.MODELS_TO_ROUTE = router_kwargs.get(
            "models_to_route", get_models_to_route(self.dataset_df)
        )
        for model_name in self.MODELS_TO_ROUTE:
            self.dataset_df[model_name].fillna(0.0, inplace=True)

        self.router_name = f"{router}"
        cache = EmbeddingCache(
            cache_uri,
            local_cache_path=os.path.join(
                "data", f"embedding_cache_{router_kwargs['embedding_model']}.pkl"
            ),
        )
        # Add to router_name all the router_kwargs, other than cache_url, separated by |
        for key, value in router_kwargs.items():
            self.router_name += f"|{key}:{value}"

        # Remove fraction kwarg as not used by routers
        fraction = router_kwargs.pop("fraction", 0.7)
        if out_of_distribution:
            other_eval_names = self.dataset_df.eval_name.unique().tolist()
        else:
            other_eval_names = eval_names
        self.dataset_df, self.eval_df = build_train_eval_dataset(
            wanted_eval_name=eval_names[0],
            other_eval_names=other_eval_names,
            dataset_df=self.dataset_df,
            fraction=fraction,
            out_of_distribution=out_of_distribution,
        )

        if router == "knn":
            self.router = KNNRouter(
                train_file=self.dataset_df, **router_kwargs, cache=cache
            )
        elif router == "mlp":
            self.router = MLPRouter(
                train_file=self.dataset_df, **router_kwargs, cache=cache
            )
        elif router == "svm":
            self.router = SVMRouter(
                train_file=self.dataset_df, **router_kwargs, cache=cache
            )

    @method()
    def batch_route_prompts(self, prompts, **kwargs):
        return self.router.batch_route_prompts(prompts, **kwargs)

    @method()
    def return_eval_routing(self, eval_names, **kwargs):
        # Select the eval names in the test dataframe, return it and sample_id
        eval_df = self.eval_df[self.eval_df.eval_name.isin(eval_names)]
        eval_df["router"] = self.router.batch_route_prompts(
            eval_df.prompt.values, **kwargs
        )
        return eval_df[["sample_id", "router"]]
