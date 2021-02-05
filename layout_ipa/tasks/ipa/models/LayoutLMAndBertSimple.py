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

        self.layout_lm_config = AutoConfig.for_model(
            layout_lm_config_model_type, **layout_lm_config
        )
        self.bert_config = AutoConfig.for_model(bert_config_model_type, **bert_config)
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
        output["layout_lm"] = self.layout_lm_config.to_dict()
        output["bert"] = self.bert_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class LayoutLMAndBertSimple(PreTrainedModel):
    config_class = LayoutLMAndBertSimpleConfig
    base_model_prefix = "layout_lm_bert"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

        self.model_instruction = AutoModel.from_pretrained(
            BERT_MODEL, config=config.bert_config
        )

        self.model_ui = AutoModel.from_pretrained(
            LAYOUT_LM_MODEL, config=config.layout_lm_config
        )

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        # self.bidaf_layer = BidafAttn(768)
        self.linear_layer1 = nn.Linear(768 * 4, 2)
        # self.linear_layer1 = nn.Linear(768 * 4, 1)
        self.linear_layer2 = nn.Linear(512, 2)

        self.act = nn.LogSoftmax(dim=1)

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
        instruction_representation = output_instruction_model[1]
        output1 = self.dropout1(instruction_representation)
        output_ui_model = self.model_ui(**input_ui)

        ui_representation = output_ui_model[1]
        output2 = self.dropout2(ui_representation)
        # both_representations = torch.cat(
        #     (ui_representation, instruction_representation), dim=1
        # )

        # print(both_representations.shape)
        both_representations = torch.cat(
            [output1, output2, torch.abs(output1 - output2), output1 * output2], dim=1
        )
        both_representations = self.linear_layer1(both_representations)
        # both_representations = self.dropout2(both_representations)
        # output = self.linear_layer2(both_representations)

        # output = output.view(-1, 261)

        output = self.act(both_representations)
        return output

    def save_pretrained(self, save_directory):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.

        Arguments:
            save_directory (:obj:`str`):
                Directory to which to save. Will be created if it doesn't exist.
        """
        if os.path.isfile(save_directory):
            logger.error(
                "Provided path ({}) should be a directory, not a file".format(
                    save_directory
                )
            )
            return
        os.makedirs(save_directory, exist_ok=True)

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        state_dict = model_to_save.state_dict()

        # Handle the case where some state_dict keys shouldn't be saved
        if self.keys_to_never_save is not None:
            state_dict = {
                k: v for k, v in state_dict.items() if k not in self.keys_to_never_save
            }

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        if getattr(self.config, "xla_device", False) and is_torch_tpu_available():
            import torch_xla.core.xla_model as xm

            if xm.is_master_ordinal():
                # Save configuration file
                model_to_save.config.save_pretrained(save_directory)
            # xm.save takes care of saving only from master
            xm.save(state_dict, output_model_file)
        else:
            model_to_save.config.save_pretrained(save_directory)
            torch.save(state_dict, output_model_file)

        logger.info("Model weights saved in {}".format(output_model_file))
