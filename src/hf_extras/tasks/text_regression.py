"""Text Regression Templates

Task templates for text regression problems. This templates are used to
standardize `datasets.Dataset` column names and types for training.

Note:
    Since HuggingFace's does not provide task templates for regression,
    this module implements those templates. The original `TextClassification`
    template forces the label to be a `ClassLabel` feature and raises an error
    when receiving something different (integer or floats).

    Implementation based on HuggingFace's text classification template:

    https://github.com/huggingface/datasets/blob/main/src/datasets/tasks/text_classification.py

"""
import copy
from dataclasses import dataclass
from datasets import Features, TaskTemplate, Value
from pyarrow import types
from typing import ClassVar, Dict, Union


@dataclass(frozen=True)
class TextRegression(TaskTemplate):
    """Dataset casting for single sentences tasks."""
    task: str = "text-regression"
    text_column: str = "text"
    label_column: str = "label"

    input_schema: ClassVar[Features] = Features({
        "text": Value("string")
    })
    label_schema: ClassVar[Features] = Features({
        "label": Union[int, float]
    })

    def align_with_features(self, features):
        if self.label_column not in features:
            msg = f"Column {self.label_column} is not present in features."
            raise ValueError(msg)

        label = features[self.label_column]
        if isinstance(label, Value):
            is_int = types.is_integer(label.pa_type)
            is_float = types.is_floating(label.pa_type)
            if not (is_int or is_float):
                raise ValueError((
                    f"Target column `{self.label_column}` "
                    "must be int or float."))
        else:
            raise ValueError((
                f"Target column `{self.label_column}` "
                "must be a `Value` feature."
            ))

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
class TextPairRegression(TaskTemplate):
    """Dataset casting for pair of sentences tasks."""
    task: str = "text-pair-regression"
    text_column: str = "text"
    text_pair_column: str = "text_pair"
    label_column: str = "label"

    input_schema: ClassVar[Features] = Features({
        "text": Value("string"),
        "text_pair": Value("string")
    })
    label_schema: ClassVar[Features] = Features({
        "label": Union[int, float]
    })

    def align_with_features(self, features):
        if self.label_column not in features:
            msg = f"Column {self.label_column} is not present in features."
            raise ValueError(msg)

        label = features[self.label_column]
        if isinstance(label, Value):
            is_int = types.is_integer(label.pa_type)
            is_float = types.is_floating(label.pa_type)
            if not (is_int or is_float):
                raise ValueError((
                    f"Target column `{self.label_column}` "
                    "must be int or float."))
        else:
            raise ValueError((
                f"Target column `{self.label_column}` "
                "must be a `Value` feature."
            ))

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
