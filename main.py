from transformers_finetuning import tasks
from transformers_finetuning import utils

import datasets


args = utils.parse_arguments()

# ------------ #
# LOAD DATASET #
# ------------ #

# optional arguments for task configuration
task_kwargs = dict()
if args.text_pair_column is not None:
    task_kwargs['text_pair_column'] = args.text_pair_column

task = tasks.get_task(
    task_name=args.task_name,
    text_column=args.text_column,
    label_column=args.label_column,
    **task_kwargs)

dataset = datasets.load_dataset(
    path=args.dataset_path,
    name=args.dataset_config,
    task=task
)
