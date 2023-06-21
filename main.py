from transformers_finetuning import tasks
from transformers_finetuning import utils

from optuna.samplers import GridSampler
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

import datasets
import evaluate
import numpy as np


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


# ------------------- #
# Optimization Metric #
# ------------------- #

metric = evaluate.load(args.metric_name)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # TODO: how to pass metric kwargs?
    results = metric.compute(
        predictions=predictions,
        references=labels)

    return results


# -------------- #
# Training Setup #
# -------------- #


# TODO: how to pass auto model kwargs?
def model_init(trial):
    auto_model = tasks.get_automodel(args.task_name)
    model = auto_model.from_pretrained(
        args.model_name,
        num_labels=args.num_labels  # TODO: this is task specific
    )
    return model


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)


# `per_device_train_batch_size' and `learning_rate' arguments
# are not passed here since they form a searching space.
training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    lr_scheduler_type=args.lr_scheduler,
    logging_strategy='epoch',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model=f"eval_{args.metric_name}",
    greater_is_better=(not args.minimize_metric),
    save_total_limit=1,
)

trainer = Trainer(
    model=None,
    args=training_args,
    model_init=model_init,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


# ------------------ #
# Optimization Setup #
# ------------------ #

# the history of metric score must saved for each trial.
# While computing the optimization metric, for each epoch,
# optuna does not save the best, but the last. That is why
# `compute_objective' return the best score so far.
_best_scores = list()

search_space = dict(
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size
)

# to try every combination in a grid search
n_trials = 1
for param in search_space.keys():
    n_trials *= len(search_space[param])


def compute_objective(metrics):
    _best_scores.append(metrics[f"eval_{args.metric_name}"])

    if args.minimize_metric:
        return min(_best_scores)
    else:
        return max(_best_scores)


def hp_space(trial):
    global _best_scores
    _best_scores = list()

    search_values = {
        "learning_rate": trial.suggest_categorical(
                "learning_rate",
                search_space["learning_rate"]),
        "per_device_train_batch_size": trial.suggest_categorical(
                "per_device_train_batch_size",
                search_space["per_device_train_batch_size"]),
    }

    return search_values


best_trial = trainer.hyperparameter_search(
    compute_objective=compute_objective,
    direction='minimize' if args.minimize_metric else 'maximize',
    backend='optuna',
    hp_space=hp_space,
    n_trials=n_trials,
    sampler=GridSampler(search_space),
    storage=f'sqlite:///{args.output_dir}/optuna_storage.db',
    study_name=f'{args.model_path}_finetuned_{args.dataset_path}',
)


print("=========================")
print("Hyperparameter Selection:")
print(f"Model: {args.model_path}")
print(f"Dataset: {args.dataset_path} ({args.dataset_config})")
print("Best Trial:")
print(f"\tRun ID: {best_trial.run_id}")
print(f"\tObjective: {best_trial.objective}")
print(f"\tHyperparameters: {best_trial.hyperparameters}")
print("=========================")
