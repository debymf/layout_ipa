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


class LayoutLMAndBertSimpleConfig(PretrainedConfig):
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


class LayoutLMAndBertSimple(PreTrainedModel):
    config_class = LayoutLMAndBertSimpleConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.model_instruction = AutoModel.from_pretrained(
            BERT_MODEL, config=config.bert
        )

        self.model_ui = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm
        )

        # self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.5)
        # self.bidaf_layer = BidafAttn(768)
        self.linear_layer1 = nn.Linear(768 * 1, 1)
        # self.linear_layer1 = nn.Linear(768 * 4, 1)
        # self.linear_layer2 = nn.Linear(512, 1)

    def forward(self, input_instructions, input_ui):

        # output_instruction_model = self.model_instruction(**input_instructions)
        # instruction_representation = output_instruction_model[0]

        # output_ui_model = self.model_ui(**input_ui)

        # ui_representation = output_ui_model[0]

        # both_representations = self.bidaf_layer(
        #     ui_representation, instruction_representation
        # )

        # both_representations = self.linear_layer1(both_representations).squeeze()

        # output = self.linear_layer2(both_representations)

        # output = output.view(-1, 261)

        output_instruction_model = self.model_instruction(**input_instructions)
        output1 = output_instruction_model[1]
        # output1 = self.dropout1(instruction_representation)
        # output_ui_model = self.model_ui(**input_ui)

        # output2 = output_ui_model[1]
        # # output2 = self.dropout2(ui_representation)
        # # both_representations = torch.cat(
        # #     (ui_representation, instruction_representation), dim=1
        # # )

        # # print(both_representations.shape)
        # both_representations = torch.cat(
        #     [output1, output2, torch.abs(output1 - output2), output1 * output2], dim=1
        # )

        output1 = F.relu(output1)
        output = self.linear_layer1(output1)

        # both_representations = self.dropout2(both_representations)
        # output = self.linear_layer2(both_representations)

        # output = output.view(-1, 261)

        # output = self.act(both_representations)
        return output

