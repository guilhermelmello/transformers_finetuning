from .base_task import BaseTask

from argparse import ArgumentParser
from transformers import AutoModelForSequenceClassification

import numpy as np


# Argument Parser for arguments required
# by TaskBase.auto_model.from_pretrained
def _arg_parser():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--num_labels",
        help="Defines the number of outputs in the last layer.",
        type=int,
        required=True
    )
    return arg_parser


class TextClassificationTask(BaseTask):
    name = "text-classification"
    auto_model = AutoModelForSequenceClassification
    input_column_names = ["text"]
    output_column_names = ["label"]

    @staticmethod
    def column_mapping(dataset, text_column, label_column):
        dataset = dataset.rename_column(text_column, "text")
        dataset = dataset.rename_column(label_column, "label")

        return dataset

    @staticmethod
    def parse_arguments(namespace):
        parser = _arg_parser()
        args = parser.parse_known_args(namespace=namespace)
        return args[0]

    @staticmethod
    def get_auto_model_arguments(args):
        return dict(
            num_labels=args.num_labels
        )

    @staticmethod
    def logits_to_outputs(logits):
        # receives a batch of logits and return the prediction index
        return np.argmax(logits, -1)


class TextPairClassificationTask(BaseTask):
    name = "text-pair-classification"
    auto_model = AutoModelForSequenceClassification
    input_column_names = ["text", "text_pair"]
    output_column_names = ["label"]

    @staticmethod
    def column_mapping(dataset, text_column, text_pair_column, label_column):
        dataset = dataset.rename_column(text_column, "text")
        dataset = dataset.rename_column(text_pair_column, "text_pair")
        dataset = dataset.rename_column(label_column, "label")

        return dataset

    @staticmethod
    def parse_arguments(namespace):
        parser = _arg_parser()
        args = parser.parse_known_args(namespace=namespace)
        return args[0]

    @staticmethod
    def get_auto_model_arguments(args):
        return dict(
            num_labels=args.num_labels
        )

    @staticmethod
    def logits_to_outputs(logits):
        # receives a batch of logits and return the prediction index
        return np.argmax(logits, -1)
