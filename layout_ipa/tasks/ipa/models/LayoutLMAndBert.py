import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoConfig
from torch.autograd import Variable
from dynaconf import settings
from transformers import PreTrainedModel
import os
import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.file_utils import WEIGHTS_NAME

torch.utils.backcompat.broadcast_warning.enabled = True
torch.set_printoptions(threshold=5000)

BERT_MODEL = "bert-base-uncased"
LAYOUT_LM_MODEL = "microsoft/layoutlm-base-uncased"


logger = logging.get_logger(__name__)


class LayoutLMAndBertConfig(PretrainedConfig):
    model_type = "layout_lm_and_bert"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "layout_lm" in kwargs and "bert" in kwargs
        ), "Layout Lm and Bert required."
        layout_lm_config = kwargs.pop("layout_lm")
        layout_lm_config_model_type = layout_lm_config.pop("model_type")

        bert_config = kwargs.pop("bert")
        bert_config_model_type = bert_config.pop("model_type")

        from transformers import AutoConfig

        self.layout_lm = AutoConfig.for_model(
            layout_lm_config_model_type, **layout_lm_config
        )
        self.bert = AutoConfig.for_model(bert_config_model_type, **bert_config)
        # self.is_encoder_decoder = True

    @classmethod
    def from_layout_lm_bert_configs(
        cls, layout_lm_config: PretrainedConfig, bert_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:

        # logger.info(
        #     "Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config"
        # )
        # decoder_config.is_decoder = True
        # decoder_config.add_cross_attention = True

        return cls(
            layout_lm=layout_lm_config.to_dict(), bert=bert_config.to_dict(), **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["layout_lm"] = self.layout_lm.to_dict()
        output["bert"] = self.bert.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class LayoutLMAndBert(PreTrainedModel):
    config_class = LayoutLMAndBertConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.model_instruction = AutoModel.from_pretrained(
            BERT_MODEL, config=config.bert
        )

        self.model_ui = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.linear_layer_instruction = nn.Linear(768, 1)
        self.linear_layer_ui = nn.Linear(768 * 2, 1)
        self.linear_layer_output = nn.Linear(768 * 2, 1)
        self.activation_ui1 = nn.Tanh()
        self.activation_ui2 = nn.Tanh()
        self.activation_instruction = nn.Tanh()

    def forward(self, input_instructions, input_ui):

        output_instruction_model = self.model_instruction.encoder(**input_instructions)

        print(output_instruction_model.shape)

        instruction_representation = output_instruction_model[1]

        output_ui_model = self.model_ui(**input_ui)
        ui_element_representation = output_ui_model[1]

        both_representations = torch.cat(
            (instruction_representation, ui_element_representation), dim=1
        )

        output = self.linear_layer_output(both_representations)

        predictions = torch.sigmoid(output)

        return output, predictions
