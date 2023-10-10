import argparse

from hf_finetuning.tasks import get_available_tasks


def get_argument_as_list_type(sep=',', item_type=str):
    """Create a list type for command line arguments.

    This method define a new type definition to be use as
    an argument type when parsing command line arguments
    that receive a list as input.

    Parameters
    ----------
    sep: str (default: ',')
        The simbol used do separate element in the input.
    item_type: Type
        Defines the type of each item in the input.
    """
    arg_type = lambda arg : [item_type(item) for item in arg.split(sep)]
    return arg_type


def get_argument_as_item_list_type(sep=',', item_type=str):
    """Create a list (or item) type for command line arguments.

    This method define a new type definition to be use as
    an argument type when parsing command line arguments
    that can receive a list or single item as input.

    Parameters
    ----------
    sep: str (default: ',')
        The simbol used do separate element in the input.
    item_type: Type
        Defines the type of each item in the input.
    """
    def arg_type(arg) :
        items = [item_type(item) for item in arg.split(sep)]
        if len(items) == 1:
            return items[0]
        return items

    return arg_type


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
        required=False,
        default=None
    )

    parser.add_argument(
        "--run_project",
        help="Name of the wandb project to report.",
        type=str,
        required=False,
        default=None
    )


    # OPTIMIZATION ARGUMENTS

    parser.add_argument(
        "--per_device_train_batch_size",
        help=(
            "Batch size per GPU/TPU core/CPU for training. If the script "
            "is using hyperparameter search, integer values can be passed "
            "separated by coma ('--per_device_train_batch_size 16,32,64')."),
        type=get_argument_as_item_list_type(sep=',', item_type=int),
        required=False,
        default=8)

    parser.add_argument(
        "--learning_rate",
        help=(
            "The initial learning rate for AdamW optimizer. If the script "
            "is using hyperparameter search, float values can be passed "
            "separated by coma ('--learning_rate 1e-5,5e-5')."),
        type=get_argument_as_item_list_type(sep=',', item_type=float),
        required=False,
        default=5e-5)

    parser.add_argument(
        "--perturbation_interval",
        help=(
            "Models will be considered for perturbation at this interval of "
            "epoch (time attribute). It incurs in checkpoint overhead, so "
            "you shouldn't set this to be too frequent. Defaults to 1."),
        type=int,
        required=False,
        default=1
    )

    parser.add_argument(
        "--burn_in_period",
        help=(
            "Models will be considered for perturbation at this interval of "
            "epoch (time attribute). It incurs in checkpoint overhead, so "
            "you shouldn't set this to be too frequent. Defaults to 1."),
        type=int,
        required=False,
        default=1
    )

    args = parser.parse_known_args()[0]
    return args
