from . import text_classification as text_classification
from . import text_regression as text_regression
from .mappers import TextMapper


_available_tasks = {
    # for text classification
    "text-classification": text_classification.TextClassification,
    "text-pair-classification": text_classification.TextPairClassification,

    # for text regression
    "text-regression": text_regression.TextRegression,
    "text-pair-regression": text_regression.TextPairRegression,
}


def get_available_tasks():
    return list(_available_tasks.keys())


def get_task(task_name, *args, **kwargs):
    if task_name not in _available_tasks:
        raise ValueError(f"'{task_name}' is not a valid task.")
    else:
        return _available_tasks[task_name](*args, **kwargs)


def get_dataset_mapper(tokenizer, text_pairs=False):
    if text_pairs:
        return TextMapper.textpair2token(tokenizer)
    else:
        return TextMapper.text2token(tokenizer)
