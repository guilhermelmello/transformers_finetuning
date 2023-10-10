"""Script for Transformers finetuning.

TODO:
- add metric extra arguemnts
- auto push to hub?
- add logging system?
"""
from dotenv import load_dotenv
from ray import tune
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

import os
import shutil
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


# path to save outputs (models)
project_dir = os.path.join(
    args.output_dir,
    args.run_project or ""
)


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
def model_init():
    task_kwargs = task.get_auto_model_arguments(args)
    model = task.auto_model.from_pretrained(
        args.model_path,
        **task_kwargs       # args defined by the task
    )
    return model


#----------------------------#
#--- FINETUNING -------------#
#----------------------------#


# Training Arguments
#--------------------

training_args = TrainingArguments(
    output_dir=project_dir,

    num_train_epochs=args.num_train_epochs,
    # per_device_train_batch_size=args.per_device_train_batch_size,   # HP search
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    lr_scheduler_type=args.lr_scheduler,
    # learning_rate=args.learning_rate,                               # HP search
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
    load_best_model_at_end=False,
    save_total_limit=1,

    report_to=args.report_to,
    run_name=args.run_name,

    # disable_tqdm=True,
    # fp16=True,                  # -> slower training
    # dataloader_num_workers=32   # -> slower training
    # push_to_hub
    # auto_find_batch_size
)

print("="*30)
print("Training Arguments:")
print(training_args)
print("="*30)


# Evaluation Metric
#-------------------

# Macro F1 for Multiclass Classification
metric_kwargs = dict()
if args.task_name in ("text-pair-classification" or "text-classification"):
    if args.num_labels > 2 and args.metric_name.upper() == "F1":
        metric_kwargs["average"] = "macro"


eval_metric = evaluate.load(args.metric_name)


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to outputs
    predictions = task.logits_to_outputs(logits)

    results = eval_metric.compute(
        predictions=predictions,
        references=labels,
        # TODO: how to pass metric kwargs?
        **metric_kwargs
    )

    return results


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    return_tensors="pt"
)

print("Data Collator:", data_collator)


# Trainer
#---------

trainer = Trainer(
    model_init=model_init,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
print("Trainer:", trainer)


# Hyperparameter Search 
#-----------------------

# assert hyperparameters are lists

lr = args.learning_rate \
    if isinstance(args.learning_rate, list) \
    else [args.learning_rate]

bsz = args.per_device_train_batch_size \
    if isinstance(args.per_device_train_batch_size, list) \
    else [args.per_device_train_batch_size]

# Using `tune.grid_search' to create the search space will
# sample every possible combination of hyperparameters.
# Also, set 'n_trials' to 1, otherwise, the entire space is
# searched n times, repeating every combinations.+

search_space = dict(
    learning_rate=tune.grid_search(lr),
    per_device_train_batch_size=tune.grid_search(bsz),
)

# Using `tune.choice' to use only hyperparameter
# values passed as argument.

mutation_space = dict(
    learning_rate=tune.choice(lr),
    per_device_train_batch_size=tune.choice(bsz),
)

scheduler = tune.schedulers.PopulationBasedTraining(
    time_attr="epoch",
    metric=f"eval_{args.metric_name}",
    mode='min' if args.minimize_metric else 'max',
    perturbation_interval=args.perturbation_interval,   # for each n epochs
    burn_in_period=args.burn_in_period,                 # skip the first n epochs
    hyperparam_mutations=mutation_space,
)


best_trial_info = trainer.hyperparameter_search(
    compute_objective=lambda metrics: metrics[f"eval_{args.metric_name}"],
    direction='minimize' if args.minimize_metric else 'maximize',
    hp_space=lambda _: search_space,
    n_trials=1,     # when using grid_search, will repeat the search space

    backend='ray',
    scheduler=scheduler,
    keep_checkpoints_num=1,
    checkpoint_score_attr="objective",
    stop={"training_iteration": args.num_train_epochs},

    storage_path=project_dir,
    log_to_file=True,
    name=args.run_name or "ray",
)


print("=========================")
print("Hyperparameter Selection:")
print(f"Model: {args.model_path}")
if args.dataset_config is None:
    print(f"Dataset: {args.dataset_path}")
else:
    print(f"Dataset: {args.dataset_path} ({args.dataset_config})")
print("Best Trial:")
print(f"\tRun ID: {best_trial_info.run_id}")
print(f"\tObjective: {best_trial_info.objective}")
print(f"\tHyperparameters: {best_trial_info.hyperparameters}")
print("=========================\n\n")


# Keep only the best trial

best_trial = best_trial_info.run_summary.get_best_trial(
    metric=f"eval_{args.metric_name}",
    mode='min' if args.minimize_metric else 'max'
)

print("Removing extra ray tune trials...")
best_trial_path = os.path.abspath(best_trial.path)
ray_path = os.path.dirname(best_trial_path)

for fname in os.listdir(ray_path):
    fpath = os.path.join(ray_path, fname)
    if fpath == best_trial_path:
        print(fname, ": will not be removed")
        continue

    print(fname, ": cleanup start")
    if os.path.isfile(fpath):
        os.remove(fpath)
    if os.path.isdir(fpath):
        shutil.rmtree(fpath)

print("Finetuning: DONE")
