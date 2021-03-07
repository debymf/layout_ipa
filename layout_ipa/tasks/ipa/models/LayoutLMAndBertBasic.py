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


class LayoutLMAndBertBasicConfig(PretrainedConfig):
    model_type = "layout_lm_and_bert"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assert (
        #     "layout_lm" in kwargs and "bert" in kwargs
        # ), "Layout Lm and Bert required."
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
        cls, layout_lm_config: PretrainedConfig, bert_config=PretrainedConfig, **kwargs
    ) -> PretrainedConfig:

        return cls(
            bert=bert_config.to_dict(), layout_lm=layout_lm_config.to_dict(), **kwargs
        )

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["layout_lm"] = self.layout_lm.to_dict()
        output["bert"] = self.bert_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class LayoutLMAndBertBasic(PreTrainedModel):
    config_class = LayoutLMAndBertBasicConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, screen_agg, combine_output, dropout, *args, **kwargs):
        super().__init__(config)

        self.screen_agg = screen_agg
        self.combine_output = combine_output
        self.model_ui_element = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )
        self.model_screen = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )
        self.instruction = AutoModel.from_pretrained(BERT_MODEL, config=config.bert)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)
        self.dropout4 = nn.Dropout(p=dropout)

        self.linear_layer_instruction = nn.Linear(768 * 3, 1)
        self.linear_screen_fc = nn.Linear(768 * 5, 768)
        self.linear_screen = nn.Linear(256 * 5, 768)
        self.linear_ui_element = nn.Linear(768, 768)
        self.linear_combine = nn.Linear(768 * 4, 128)
        self.linear_combine_simple = nn.Linear(768, 128)
        self.linear_combine_double = nn.Linear(768 * 2, 128)
        self.linear_layer_ui = nn.Linear(768 * 5, 768)
        self.linear_layer_output = nn.Linear(768 * 3, 1)
        self.activation_ui1 = nn.Tanh()
        self.activation_ui2 = nn.Tanh()
        self.activation_instruction = nn.Tanh()

        # self.linear_layer1 = nn.Linear(768 * 4, 1)
        # self.linear_layer2 = nn.Linear(512, 1)

    def forward(self, screen, instruction, ui_element):

        instruction_embedding = self.bert(**instruction)[1]
        ui_embedding = self.model_ui_element(**ui_element)[1]
        screen_embedding = self.model_screen(**screen)[1]

        output = torch.cat([instruction_embedding, ui_embedding, screen_embedding])

        output = self.linear_layer_output(output)
        return output

