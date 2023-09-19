from ..utils import classproperty
from argparse import Namespace
from datasets import Dataset, DatasetDict
from numbers import Number
from transformers import AutoModel
from typing import List, Union, Mapping


_attribute_error_msg = "Task '{}' is missing the attribute '{}'."
_method_error_msg = "Task '{}' is missing the method '{}'."


class BaseTask:
    @classproperty
    def name(cls) -> str:
        raise NotImplementedError(
            _attribute_error_msg.format(cls.__name__, "name")
        )
    
    @classproperty
    def column_names(cls) -> List[str]:
        return cls.input_column_names + cls.output_column_names
    
    @classproperty
    def output_column_names(cls) -> List[str]:
        raise NotImplementedError(
            _attribute_error_msg.format(cls.__name__, "output_column_names")
        )
    
    @classproperty
    def input_column_names(cls) -> List[str]:
        raise NotImplementedError(
            _attribute_error_msg.format(cls.__name__, "input_column_names")
        )
    
    @classproperty
    def auto_model(cls) -> AutoModel:
        raise NotImplementedError(
            _attribute_error_msg.format(cls.__name__, "auto_model")
        )
    
    @classmethod
    def column_mapping(cls, *args, **kwargs) -> Union[Dataset, DatasetDict]:
        raise NotImplementedError(
            _method_error_msg.format(cls.__name__, "column_mapping")
        )
    
    @classmethod
    def parse_arguments(cls, namespace: Namespace) -> Namespace:
        raise NotImplementedError(
            _method_error_msg.format(cls.__name__, "parse_arguments")
        )

    @classmethod
    def get_auto_model_arguments(cls, namespace: Namespace) -> Mapping[str, Union[str, Number]]:
        raise NotImplementedError(
            _method_error_msg.format(cls.__name__, "get_auto_model_arguments")
        )
    
    @staticmethod
    def logits_to_outputs(logits, batched: bool =True):
        """Convert model logits to model outputs.
        
        This method is task specific and must be overrided when needed.
        By default, no processing is made on model logits. For example,
        classification problems need to convert logits into predictions
        by selecting the index of the greatest logit.
        """
        return logits

    @classmethod
    def prepare_dataset(
        cls,
        dataset: Union[Dataset, DatasetDict],
        **kwargs
    ) -> Union[Dataset, DatasetDict]:
        # rename and select columns for task
        dataset = cls.column_mapping(dataset, **kwargs)
        dataset = dataset.select_columns(cls.column_names)

        return dataset
