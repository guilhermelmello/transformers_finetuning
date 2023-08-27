"""Text Classification Templates

Text classification task templates to standardize `datasets.Dataset`
column names and types for training.
"""
import copy
from dataclasses import dataclass
from datasets import ClassLabel, Features, TaskTemplate, Value
from typing import ClassVar, Dict


@dataclass(frozen=True)
class TextClassification(TaskTemplate):
    """Dataset casting for single sentences tasks.

    Reimplementation of the original text classification template:
    https://github.com/huggingface/datasets/blob/main/src/datasets/tasks/text_classification.py
    """
    task: str = "text-classification"
    text_column: str = "text"
    label_column: str = "label"

    input_schema: ClassVar[Features] = Features({
        "text": Value("string")
    })
    label_schema: ClassVar[Features] = Features({
        "label": ClassLabel
    })

    def align_with_features(self, features):
        if self.label_column not in features:
            msg = f"Column {self.label_column} is not present in features."
            raise ValueError(msg)
        if not isinstance(features[self.label_column], ClassLabel):
            msg = f"Column {self.label_column} is not a ClassLabel."
            raise ValueError(msg)

        # update label schema to reflect label feature
        label_schema = self.label_schema.copy()
        label_schema["label"] = features[self.label_column]

        # updated task template
        task_template = copy.deepcopy(self)
        task_template.__dict__['label_schema'] = label_schema

        return task_template

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.text_column: "text",
            self.label_column: "label"
        }


@dataclass(frozen=True)
class TextPairClassification(TaskTemplate):
    """Dataset casting for pair of sentences tasks.

    HuggingFace's `datasets` does not provide a `TextClassification` task
    template that standardize pairs of sentences as input. This template is
    an extension of the original `TextClassification` for pair of sentences.


    """
    task: str = "text-pair-classification"
    text_column: str = "text"
    text_pair_column: str = "text_pair"
    label_column: str = "label"

    input_schema: ClassVar[Features] = Features({
        "text": Value("string"),
        "text_pair": Value("string")
    })
    label_schema: ClassVar[Features] = Features({
        "label": ClassLabel
    })

    def align_with_features(self, features):
        if self.label_column not in features:
            msg = f"Column {self.label_column} is not present in features."
            raise ValueError(msg)
        if not isinstance(features[self.label_column], ClassLabel):
            msg = f"Column {self.label_column} is not a ClassLabel."
            raise ValueError(msg)

        # update label schema to reflect label feature
        label_schema = self.label_schema.copy()
        label_schema["label"] = features[self.label_column]

        # updated task template
        task_template = copy.deepcopy(self)
        task_template.__dict__['label_schema'] = label_schema

        return task_template

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.text_column: "text",
            self.text_pair_column: "text_pair",
            self.label_column: "label"
        }
