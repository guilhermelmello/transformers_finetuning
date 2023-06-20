import argparse

from .tasks import get_available_tasks


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Automatic finetuning for Transformer models.")

    # DATASET ARGUMENTS
    parser.add_argument(
        "--dataset_path",
        help="Path or name of the dataset. Used by `datasets.load_dataset'",
        type=str,
        default=None,
        required=True)

    parser.add_argument(
        "--dataset_config",
        help=("Name of the dataset config. Used by `datasets.load_dataset'"),
        type=str,
        default=None)

    parser.add_argument(
        "--text_column",
        help="Name of the text column in the dataset.",
        type=str,
        default="text")

    parser.add_argument(
        "--text_pair_column",
        help="Name of the text column pair. Used on pair of sequences tasks.",
        type=str,
        default=None)

    parser.add_argument(
        "--label_column",
        help="Name of the target column.",
        type=str,
        default="label")

    parser.add_argument(
        "--dataprep_batch_size",
        help=("Batch size for dataset tokenization."
              "For training, use `batch_size'"),
        type=int,
        default=1000)

    parser.add_argument(
        "--task_name",
        help="The task defining the a Machine Learning problem/setup.",
        type=str,
        default="text-classification",
        choices=get_available_tasks())

    # MODEL ARGUMENTS
    parser.add_argument(
        "--model_path",
        help="",
        type=str,
        default=None,
        required=True)

    args = parser.parse_args()
    return args
