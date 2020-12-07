# Code adapted from https://github.com/microsoft/unilm/blob/master/layoutlm/layoutlm/data/funsd.py

from prefect import Task
from loguru import logger
from dynaconf import settings
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer
from torch.utils.data import Dataset
import torch

tokenizer_model = settings["layout_lm_base"]


class PrepareRicoTransformerSelect(Task):
    # def run(self, input_data, bert_model, num_choices=261,largest=512):
    #     logger.info("*** Preprocessing Data for Transformer-Based ***")
    #     tokenizer= AutoTokenizer.from_pretrained(bert_model)
    #     tokenizer_instruction = BertTokenizer.from_pretrained("bert-base-uncased")
    #     entries = dict()
        
    #     for id_d, content in tqdm(input_data.items()):
    #         ui_elements = []
    #         encoding_instruction = []
    #         for id_ui, ui_element in content["ui"].items():    
    #             ui_elements.append(ui_element["text"])

    #         if len(ui_elements)<num_choices:
    #             ui_elements.extend([""]*(num_choices-len(ui_elements)))

            
    #         instruction_list = [content["instruction"]] * num_choices


            

    #         encoded_element = tokenizer_instruction(
    #                 instruction_list, ui_elements, padding="max_length", max_length=largest, truncation=True
    #         )
    #         print(encoded_element)
    #         input()

    #         entries[id_d] = {
    #             "input_ids": encoded_element["input_ids"],
    #             "att_mask": encoded_element["attention_mask"],
    #             "token_ids": encoded_element["token_type_ids"],
    #             "label": content["label"],
    #         }

    #     return TorchDataset(entries)


    def run(self, input_data, bert_model, largest=512):
        logger.info("*** Preprocessing Data for Transformer-Based ***")
        tokenizer= AutoTokenizer.from_pretrained(bert_model)
        tokenizer_instruction = BertTokenizer.from_pretrained("bert-base-uncased")
        entries = dict()
        
        for id_d, content in tqdm(input_data.items()):
            ui_elements = ""
            for id_ui, ui_item in content["ui"].items():    
                if ui_elements == "":
                    ui_elements = ui_item["text"]
                else:
                    ui_elements = ui_elements + " [SEP] " + ui_item["text"]


            encoded_element = tokenizer_instruction.encode_plus(
                    content["instruction"], ui_elements, padding="max_length", max_length=largest, truncation=True
            )


            entries[id_d] = {
                "input_ids": encoded_element["input_ids"],
                "att_mask": encoded_element["attention_mask"],
                "token_ids": encoded_element["token_type_ids"],
                "label": content["label"],
            }

        return TorchDataset(entries)

class TorchDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = list(dataset.values())
        self.keys = list(dataset.keys())

    def __getitem__(self, index):
        instance = self.dataset[index]
        return (
            torch.LongTensor(instance["input_ids"]),
            torch.LongTensor(instance["att_mask"]),
            torch.LongTensor(instance["token_ids"]),
            instance["label"],
            index,
        )

    def get_id(self, index):
        return self.keys[index]

    def __len__(self):
        return len(self.dataset)


class MultipleChoiceInputFeatures(object):
    """
    Class representing multiple-choice question.
    Credit to
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_swag.py
    """
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label