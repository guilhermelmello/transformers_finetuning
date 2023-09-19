from .text_classification import TextClassificationTask
from .text_classification import TextPairClassificationTask
from .text_regression import TextRegressionTask
from .text_regression import TextPairRegressionTask


_tasks = {
    # for text classification
    TextClassificationTask.name: TextClassificationTask,
    TextPairClassificationTask.name: TextPairClassificationTask,

    # for text regression
    TextRegressionTask.name: TextRegressionTask,
    TextPairRegressionTask.name: TextPairRegressionTask,
}


def get_available_tasks():
    return list(_tasks.keys())


def get_task(task_name):
    return _tasks[task_name]
