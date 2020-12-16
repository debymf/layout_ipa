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


# class LayoutIpa(nn.Module):
#     def __init__(
#         self, batch_size,
#     ):
#         super(LayoutIpa, self).__init__()

#         bert_config = AutoConfig.from_pretrained(BERT_MODEL)
#         self.model_instruction = AutoModel.from_pretrained(
#             BERT_MODEL, config=bert_config
#         )

#         layout_lm_config = LayoutlmConfig.from_pretrained(LAYOUT_LM_MODEL)
#         self.model_ui = LayoutlmModel.from_pretrained(
#             LAYOUT_LM_MODEL, config=layout_lm_config
#         )

#         self.dropout1 = nn.Dropout(p=0.5)
#         self.dropout2 = nn.Dropout(p=0.5)
#         self.bidaf_layer = BidafAttn(768)
#         self.linear_layer1 = nn.Linear(768 * 4, 20)
#         # self.linear_layer1 = nn.Linear(768 * 4, 1)
#         self.linear_layer2 = nn.Linear(512, 20)

#     def forward(self, input_instructions, input_ui):

#         # output_instruction_model = self.model_instruction(**input_instructions)
#         # instruction_representation = output_instruction_model[0]

#         # output_ui_model = self.model_ui(**input_ui)

#         # ui_representation = output_ui_model[0]

#         # both_representations = self.bidaf_layer(
#         #     ui_representation, instruction_representation
#         # )

#         # both_representations = self.linear_layer1(both_representations).squeeze()

#         # output = self.linear_layer2(both_representations)

#         # output = output.view(-1, 261)

#         output_instruction_model = self.model_instruction(**input_instructions)
#         instruction_representation = output_instruction_model[1]
#         output1 = self.dropout1(instruction_representation)
#         output_ui_model = self.model_ui(**input_ui)

#         ui_representation = output_ui_model[1]
#         output2 = self.dropout2(ui_representation)
#         # both_representations = torch.cat(
#         #     (ui_representation, instruction_representation), dim=1
#         # )

#         # print(both_representations.shape)
#         both_representations = torch.cat(
#             [output1, output2, torch.abs(output1 - output2), output1 * output2], dim=1
#         )
#         both_representations = self.linear_layer1(both_representations)
#         # both_representations = self.dropout2(both_representations)
#         # output = self.linear_layer2(both_representations)

#         # output = output.view(-1, 261)

#         return both_representations


class LayoutIpa(nn.Module):
    def __init__(
        self, batch_size, model_instruction, model_ui
    ):
        super(LayoutIpa, self).__init__()


        self.model_instruction = model_instruction
        self.model_ui = model_ui
        # bert_config = AutoConfig.from_pretrained(BERT_MODEL)
        # self.model_instruction = AutoModel.from_pretrained(
        #     BERT_MODEL, config=bert_config
        # )

        # layout_lm_config = LayoutlmConfig.from_pretrained(LAYOUT_LM_MODEL)
        # self.model_ui = LayoutlmModel.from_pretrained(
        #     LAYOUT_LM_MODEL, config=layout_lm_config
        # )

        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.bidaf_layer = BidafAttn(768)
        self.linear_layer1 = nn.Linear(768 * 2, 1)
        # self.linear_layer1 = nn.Linear(768 * 4, 1)
        self.linear_layer2 = nn.Linear(512, 20)

    def forward(self, input_instructions, input_ui):

        num_choices = input_ui["input_ids"].shape[1]
        output_instruction_model = self.model_instruction(**input_instructions)

        instruction_representation = output_instruction_model[1]
        output1 = self.dropout1(instruction_representation)

        input_ui["input_ids"] = input_ui["input_ids"].view(
            -1, input_ui["input_ids"].size(-1)
        )
        input_ui["position_ids"] = input_ui["position_ids"].view(
            -1, input_ui["position_ids"].size(-1)
        )
        input_ui["token_type_ids"] = input_ui["token_type_ids"].view(
            -1, input_ui["token_type_ids"].size(-1)
        )
        input_ui["bbox"] = input_ui["bbox"].view(-1, input_ui["bbox"].size(-2), 4)

        
        output_ui_model = self.model_ui(**input_ui)
        ui_representation = output_ui_model[1]
        ui_representation = self.dropout2(ui_representation)

        instruction_representation = instruction_representation.repeat(num_choices,1)


        both_representations = torch.cat((ui_representation, instruction_representation), dim=1)
        # both_representations = torch.cat(
        #     [output1, output2, torch.abs(output1 - output2), output1 * output2], dim=1
        # )
        #both_representations = output1 * output2
        

        both_representations = self.linear_layer1(both_representations)
        both_representations = self.dropout2(both_representations)

        both_representations = both_representations.view(-1, num_choices)


        # output = self.linear_layer2(both_representations)

        # output = output.view(-1, 261)

        return both_representations
