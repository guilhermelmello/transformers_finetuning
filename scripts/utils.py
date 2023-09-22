import argparse

from hf_finetuning.tasks import get_available_tasks


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Automatic finetuning for Transformer models."
    )


    # TASK ARGUMENTS

    parser.add_argument(
        "--task_name",
        help="The task defining the a Machine Learning problem/setup.",
        type=str,
        required=True,
        choices=get_available_tasks()
    )

    # LOAD DATASET

    parser.add_argument(
        "--dataset_path",
        help="Path or name of the dataset. Used by `datasets.load_dataset'",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--dataset_config",
        help="Name of the dataset config. Used by `datasets.load_dataset'",
        type=str,
        default=None
    )

    # DATASET PREPARATION

    parser.add_argument(
        "--text_column",
        help="Name of the input text column in the dataset.",
        type=str,
        default="text"
    )

    parser.add_argument(
        "--text_pair_column",
        help=(
            "Name of the text pair column. "
            "Used by tasks on pair of sequences as inputs."),
        type=str,
        default=None
    )

    parser.add_argument(
        "--label_column",
        help="Name of the target column (output).",
        type=str,
        default="label"
    )

    parser.add_argument(
        "--data_map_bsz",
        help=(
            "Batch size for dataset tokenization. For training batch "
            "size, see `per_device_train_batch_size' argument"),
        type=int,
        default=1000
    )


    # MODEL ARGUMENTS

    parser.add_argument(
        "--model_path",
        help="Path or name of the model and tokenizer.",
        type=str,
        required=True
    )


    # TRAINING ARGUMENTS

    parser.add_argument(
        "--output_dir",
        help=(
            "The output directory where the checkpoints will be saved. "
            "Defaults to current directory."),
        type=str,
        default="."
    )

    parser.add_argument(
        "--num_train_epochs",
        help="Total number of training epochs to perform.",
        type=int,
        default=3
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        help="Batch size per GPU/TPU core/CPU for training.",
        type=int,
        default=8
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        help="Batch size per GPU/TPU core/CPU for evaluation.",
        type=int,
        default=8
    )

    parser.add_argument(
        "--lr_scheduler",
        help="The scheduler type to use. Default: 'linear'.",
        type=str,
        default='linear'
    )

    parser.add_argument(
        "--learning_rate",
        help="The initial learning rate for AdamW optimizer.",
        type=float,
        default=5e-5
    )

    parser.add_argument(
        "--weight_decay",
        help=(
            "The weight decay to apply (if not zero) to all layers "
            "except all bias and LayerNorm weights in AdamW optimizer"),
        type=float,
        default=0
    )

    parser.add_argument(
        "--adam_beta1",
        help="The beta1 hyperparameter for the AdamW optimizer.",
        type=float,
        default=0.9
    )

    parser.add_argument(
        "--adam_beta2",
        help="The beta2 hyperparameter for the AdamW optimizer.",
        type=float,
        default=0.999
    )

    parser.add_argument(
        "--adam_epsilon",
        help="The epsilon hyperparameter for the AdamW optimizer.",
        type=float,
        default=1e-8
    )

    parser.add_argument(
        "--warmup_ratio",
        help=(
            "Ratio of total training steps used for "
            "a linear warmup from 0 to learning_rate."),
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--warmup_steps",
        help=(
            "Number of steps used for a linear warmup from 0 to "
            "learning_rate. Overrides any effect of warmup_ratio."),
        type=int,
        default=0
    )

    parser.add_argument(
        "--metric_name",
        help="Specify the metric used to compare different models.",
        type=str,
        required=True
    )

    parser.add_argument(
        "--minimize_metric",
        help=(
            "Specify if better models have lower metric "
            "value. By default, greater is better."),
        action="store_true"
    )

    parser.add_argument(
        "--report_to",
        help=(
            "The name of the integration to report the results and logs to. See"
            "Transformers.TrainingArgument for supported platforms. By default,"
            " no report is enabled."),
        type=str,
        default="none"
    )

    parser.add_argument(
        "--run_name",
        help="A descriptor for the run. Typically used by wandb for logging.",
        type=str,
        required=False
    )

    parser.add_argument(
        "--run_project",
        help="Name of the wandb project to report.",
        type=str,
        required=False,
        default=None
    )


    # OPTIMIZATION ARGUMENTS
    # parser.add_argument(
    #     "--metric_name",
    #     help="Path or name of the metric to optimize.",
    #     type=str,
    #     required=True)

    # parser.add_argument(
    #     "--minimize_metric",
    #     help=("To minimize the optimization metric (`--metric_name'). "
    #           "By default, the metric will be maximized."),
    #     action="store_true")

    # parser.add_argument(
    #     "--per_device_train_batch_size",
    #     help=("Batch size per GPU/TPU core/CPU for training. "
    #           "Integer values must be passed separated by coma. "
    #           "Example: '--per_device_train_batch_size 16,32,64'."),
    #     type=lambda s: [int(item) for item in s.split(',')],
    #     required=True)

    # parser.add_argument(
    #     "--learning_rate",
    #     help=("The initial learning rate for AdamW optimizer. "
    #           "Float values must be passed separated by coma. "
    #           "Example: '--learning_rate 1e-5,5e-5'."),
    #     type=lambda s: [float(item) for item in s.split(',')],
    #     required=True)

    args = parser.parse_known_args()[0]
    return args
