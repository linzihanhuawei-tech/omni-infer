import json
import os
import warnings
from typing import Optional, Union

import torch
from transformers import AutoConfig, PretrainedConfig, modeling_utils
from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import (
    Qwen2Config,
)

from omni.speculative_train.models.draft.qwen2_eagle import EagleQwen2ForCausalLM

class AutoEagleDraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        Qwen2Config: EagleQwen2ForCausalLM,
    }

    @classmethod
    def from_config(cls, config: PretrainedConfig, **config_kwargs):
        """
        This class method takes a configuration object and create its model based on the
        _model_mapping class variable.

        Args:
            config (PretrainedConfig): A configuration object.

        Returns:
            A model instance.
        """
        # get the model class from the
        _model_cls = cls._model_mapping[type(config)]
        return _model_cls(config, **config_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike[str]],
        *model_args,
        **kwargs,
    ):
        original_warn = modeling_utils.logger.warning

        def filtered_warning(msg):
            if "embed_tokens.weight" in str(msg) and "initialized" in str(msg):
                return
            original_warn(msg)

        modeling_utils.logger.warning = filtered_warning

        try:
            model = super().from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        finally:
            modeling_utils.logger.warning = original_warn

        return model

class AutoDraftModelConfig:

    _config_mapping = {
        "EagleQwen2ForCausalLM": Qwen2Config,
        "Qwen2ForCausalLM": Qwen2Config,
    }

    @classmethod
    def from_file(cls, config_path: str):
        """
        This class method takes a configuration file path and create its configuration object based on the
        _config_mapping class variable.

        Args:
            config_path (str): A path to a configuration file.

        Returns:
            A configuration object.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        if "tie_word_embeddings" in config:
            print("Set draft model tie_word_embeddings to False")
            config["tie_word_embeddings"] = False

        # check for architectures
        architectures = config.get("architectures", None)

        if architectures is None:
            raise ValueError("No architectures found in the config file")

        if len(architectures) != 1:
            raise ValueError("Only one architecture is supported")

        architecture = architectures[0]

        if architecture not in cls._config_mapping:
            raise ValueError(f"Architecture {architecture} not supported")

        return cls._config_mapping[architecture].from_dict(config)
