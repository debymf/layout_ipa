import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoConfig
from torch.autograd import Variable
from dynaconf import settings
from .layoutlm import LayoutlmConfig, LayoutlmEmbeddings, LayoutlmModel
from .bidaf import BidafAttn
torch.utils.backcompat.broadcast_warning.enabled = True
torch.set_printoptions(threshold=5000)

BERT_MODEL = "bert-base-uncased"
LAYOUT_LM_MODEL = settings["layout_lm_base"]


class LayoutIpa(nn.Module):
    def __init__(
        self, batch_size,
    ):
        super(LayoutIpa, self).__init__()

        bert_config = AutoConfig.from_pretrained(BERT_MODEL)
        self.model_instruction = AutoModel.from_pretrained(
            BERT_MODEL, config=bert_config
        )

        layout_lm_config = LayoutlmConfig.from_pretrained(LAYOUT_LM_MODEL)
        self.model_ui = LayoutlmModel.from_pretrained(
            LAYOUT_LM_MODEL, config=layout_lm_config
        )

        self.bidaf_layer = BidafAttn(768)
        self.linear_layer1 = nn.Linear(768 * 4, 1)
        self.linear_layer2 = nn.Linear(512, 261)

    def forward(self, input_instructions, input_ui):

        output_instruction_model = self.model_instruction(**input_instructions)
        instruction_representation = output_instruction_model[1]


        output_ui_model = self.model_ui(**input_ui)


        ui_representation = output_ui_model[1]


        #both_representations = self.bidaf_layer(ui_representation, instruction_representation)

        both_representations = self.linear_layer1(both_representations).squeeze()

        output = self.linear_layer2(both_representations)

        output = output.view(-1, 261)
        
        

        return output

