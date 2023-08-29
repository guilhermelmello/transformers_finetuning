from . import text_classification
from . import text_regression

from transformers import AutoModelForSequenceClassification


_available_tasks = {
    # for text classification
    "text-classification": text_classification.TextClassification,
    "text-pair-classification": text_classification.TextPairClassification,

    # for text regression
    "text-regression": text_regression.TextRegression,
    "text-pair-regression": text_regression.TextPairRegression,
}

_auto_models = {
    # for text classification
    "text-classification": AutoModelForSequenceClassification,
    "text-pair-classification": AutoModelForSequenceClassification,

    # for text regression TODO
    # "text-regression":
    # "text-pair-regression":
}


def get_available_tasks():
    return list(_available_tasks.keys())


def get_task(task_name, *args, **kwargs):
    if task_name not in _available_tasks:
        raise ValueError(f"'{task_name}' is not a valid task.")
    else:
        return _available_tasks[task_name](*args, **kwargs)


def get_automodel(task_name):
    if task_name not in _auto_models:
        raise ValueError(f"Could not find an AutoModel for '{task_name}'.")
    else:
        return _auto_models[task_name]
