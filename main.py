from transformers_finetuning import tasks
from transformers_finetuning import utils

from transformers import AutoTokenizer

import datasets


args = utils.parse_arguments()

# ------------ #
# LOAD DATASET #
# ------------ #

# optional arguments for task configuration
task_kwargs = dict()
if args.text_pair_column is not None:
    task_kwargs['text_pair_column'] = args.text_pair_column

# map dataset's columns to predefined names
task = tasks.get_task(
    task_name=args.task_name,
    text_column=args.text_column,
    label_column=args.label_column,
    **task_kwargs)

# load a hugginface's dataset
dataset = datasets.load_dataset(
    path=args.dataset_path,
    name=args.dataset_config,
    task=task
)


# -------------------- #
# Dataset Tokenization #
# -------------------- #

# load tokenizer and map function
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
to_tokens = tasks.get_dataset_mapper(
    tokenizer=tokenizer,
    text_pairs=args.text_pair_column)

# text columns to drop
if args.text_pair_column is not None:
    drop_columns = ['text', 'text_pair']
else:
    drop_columns = ['text']

# dataset tokenization
dataset = dataset.map(
    function=to_tokens,
    batched=True,
    batch_size=args.dataprep_batch_size,
    remove_columns=drop_columns
)
