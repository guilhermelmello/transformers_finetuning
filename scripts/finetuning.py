"""Script for Transformers finetuning.

TODO:
- add metric extra arguemnts
- add support for wandb report
- auto push to hub?
- add logging system
"""
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import datasets
import evaluate
import hf_finetuning as hff
import hf_finetuning.tasks as hff_tasks
import numpy as np

import os
import utils
import wandb


#----------------------------#
#--- PARSE ARGUMENTS --------#
#----------------------------#


# parse default arguments
args = utils.parse_arguments()

# load the task config class
task = hff_tasks.get_task(task_name=args.task_name)

# parse task arguments
args = task.parse_arguments(args)


#----------------------------#
#--- LOGGING SETUP ----------#
#----------------------------#


load_dotenv()


# wandb setup
if args.report_to == "wandb":
    wandb.login(key=os.environ["WANDB_API_TOKEN"])

    if args.run_project is not None:
        os.environ["WANDB_PROJECT"] = args.run_project


#----------------------------#
#--- LOAD DATASET -----------#
#----------------------------#


# load a hugginface's dataset
dataset = datasets.load_dataset(
    path=args.dataset_path,
    name=args.dataset_config,
)

print("Loaded Dataset:")
print(dataset)


#----------------------------#
#--- DATASET PREPARATION ----#
#----------------------------#


# optional task arguments
task_kwargs = dict()

# Single sentence tasks can not receive 'text_pair_column' argument.
if args.text_pair_column is not None:
    task_kwargs['text_pair_column'] = args.text_pair_column


task = hff_tasks.get_task(task_name=args.task_name)
dataset = task.prepare_dataset(
    dataset=dataset,
    text_column=args.text_column,
    label_column=args.label_column,
    **task_kwargs
)

print("Prepared Dataset:")
print(dataset)


#----------------------------#
#--- DATASET TOKENIZATION ---#
#----------------------------#


# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# load the function to map a single sentence, or
# a pair of sentences, into a sequence of tokens.
to_tokens = hff.get_dataset_mapper(
    tokenizer=tokenizer,
    text_pairs=args.text_pair_column
)

# dataset tokenization
dataset = dataset.map(
    function=to_tokens,
    batch_size=args.data_map_bsz,
    batched=True,
)

# remove task unrelated columns
dataset = dataset.select_columns(
    tokenizer.model_input_names + task.output_column_names
)

print("Tokenized Dataset:")
print(dataset)


#----------------------------#
#--- LOAD THE MODEL ---------#
#----------------------------#


# load the task AutoModel class (AutoModelFor[TASK])
print("Auto Model:", task.auto_model)

# load a pretrained model
task_kwargs = task.get_auto_model_arguments(args)
model = task.auto_model.from_pretrained(
    args.model_path,
    **task_kwargs       # args defined by the task
)
print("Model:", model)


#----------------------------#
#--- FINETUNING -------------#
#----------------------------#


# Training Arguments
#--------------------

# `per_device_train_batch_size' and `learning_rate' arguments
# are not passed here since they form a searching space.
training_args = TrainingArguments(
    output_dir=args.output_dir,

    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    lr_scheduler_type=args.lr_scheduler,
    learning_rate=args.learning_rate,
    weight_decay=args.weight_decay,
    adam_beta1=args.adam_beta1,
    adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_epsilon,
    warmup_ratio=args.warmup_ratio,
    warmup_steps=args.warmup_steps,

    metric_for_best_model=f"eval_{args.metric_name}",
    greater_is_better=(not args.minimize_metric),

    logging_strategy='epoch',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    save_total_limit=1,

    report_to=args.report_to,
    run_name=args.run_name

    # disable_tqdm=True,
    # fp16=True,                -> training speedup?
    # dataloader_num_workers    -> training speedup?
    # push_to_hub
    # auto_find_batch_size
)

print("="*30)
print("Training Arguments:")
print(training_args)
print("="*30)


# Evaluation Metric
#-------------------

eval_metric = evaluate.load(args.metric_name)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to outputs
    predictions = task.logits_to_outputs(logits)

    results = eval_metric.compute(
        predictions=predictions,
        references=labels,
        # TODO: how to pass metric kwargs?
    )

    return results


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)

print("Data Collator:", data_collator)


# Training
#----------

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
print("Trainer:", trainer)

print("Training: Start")
trainer.train()
print("Training: Completed")
