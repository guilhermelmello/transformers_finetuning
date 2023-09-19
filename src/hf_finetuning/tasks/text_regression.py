from .base_task import BaseTask

from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification


class TextRegressionTask(BaseTask):
    name = "text-regression"
    auto_model = AutoModelForSequenceClassification
    input_column_names = ["text"]
    output_column_names = ["target"]

    @staticmethod
    def column_mapping(dataset, text_column, target_column):
        dataset = dataset.rename_column(text_column, "text")
        dataset = dataset.rename_column(target_column, "target")

        return dataset

    @staticmethod
    def parse_arguments(namespace):
        # no arguments required by this task
        return namespace

    @staticmethod
    def get_auto_model_arguments(*args, **kwargs):
        return dict(
            num_labels=1
        )


class TextPairRegressionTask(BaseTask):
    name = "text-pair-regression"
    auto_model = AutoModelForSequenceClassification
    input_column_names = ["text", "text_pair"]
    output_column_names = ["target"]

    @staticmethod
    def column_mapping(dataset, text_column, text_pair_column, target_column):
        dataset = dataset.rename_column(text_column, "text")
        dataset = dataset.rename_column(text_pair_column, "text_pair")
        dataset = dataset.rename_column(target_column, "target")

        return dataset

    @staticmethod
    def get_argument_parser(namespace):
        return namespace

    @staticmethod
    def get_auto_model_arguments(*args, **kwargs):
        return dict(
            num_labels=1
        )
