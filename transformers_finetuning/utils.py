import argparse

from .tasks import get_available_tasks


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Automatic finetuning for Transformer models.")

    # dataset arguments
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
        default="text")

    parser.add_argument(
        "--text_pair_column",
        help="Name of the text column pair. Used on pair of sequences tasks.",
        default=None)

    parser.add_argument(
        "--label_column",
        help="Name of the target column.",
        default="label")

    parser.add_argument(
        "--dataprep_batch_size",
        help=("Batch size for dataset tokenization."
              "For training, use `batch_size'"),
        default=1000)

    parser.add_argument(
        "--task_name",
        help="The task defining the a Machine Learning problem/setup.",
        default="text-classification",
        choices=get_available_tasks())

    args = parser.parse_args()
    return args
