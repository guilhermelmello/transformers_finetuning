import argparse

from ..src.hf_extras.tasks import get_available_tasks


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
        help=("Batch size for dataset tokenization. For training, "
              "see `per_device_train_batch_size' argument"),
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
        help="Path or name of the model and tokenizer.",
        type=str,
        required=True)

    # TODO: this is a task specific argument.
    parser.add_argument(
        "--num_labels",
        help="Number of outputs in the classification head.",
        type=int,
        default=None)

    # TRAINING ARGUMENTS
    parser.add_argument(
        "--output_dir",
        help=("The output directory where the checkpoints will be saved. "
              "Defaults to current directory."),
        type=str,
        default=".")

    parser.add_argument(
        "--per_device_eval_batch_size",
        help="Batch size per GPU/TPU core/CPU for evaluation.",
        type=int,
        required=True)

    parser.add_argument(
        "--num_train_epochs",
        help="Total number of training epochs to perform.",
        type=int,
        required=True
    )

    parser.add_argument(
        "--lr_scheduler",
        help="The scheduler type to use. Default: 'linear'.",
        type=str,
        default='linear'
    )

    # OPTIMIZATION ARGUMENTS
    parser.add_argument(
        "--metric_name",
        help="Path or name of the metric to optimize.",
        type=str,
        required=True)

    parser.add_argument(
        "--minimize_metric",
        help=("To minimize the optimization metric (`metric_name'). "
              "By default, the metric will be maximized."),
        action="store_true")

    parser.add_argument(
        "--per_device_train_batch_size",
        help=("Batch size per GPU/TPU core/CPU for training. "
              "Integer values must be passed separated by coma. "
              "Example: '--per_device_train_batch_size 16,32,64'."),
        type=lambda s: [int(item) for item in s.split(',')],
        required=True)

    parser.add_argument(
        "--learning_rate",
        help=("The initial learning rate for AdamW optimizer. "
              "Float values must be passed separated by coma. "
              "Example: '--learning_rate 1e-5,5e-5'."),
        type=lambda s: [float(item) for item in s.split(',')],
        required=True)

    args = parser.parse_args()
    return args
