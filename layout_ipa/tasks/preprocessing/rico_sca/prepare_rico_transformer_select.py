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
    def run(self, input_data, bert_model, num_choices=261,largest=512):
        logger.info("*** Preprocessing Data for Transformer-Based ***")
        tokenizer= AutoTokenizer.from_pretrained(bert_model)
        tokenizer_instruction = BertTokenizer.from_pretrained("bert-base-uncased")
        entries = dict()
        
        for id_d, content in tqdm(input_data.items()):
            ui_elements = []
            encoding_instruction = []
            for id_ui, ui_element in content["ui"].items():    
                ui_elements.append(ui_element["text"])

            if len(ui_elements)<num_choices:
                ui_elements.append(""*num_choices-len(ui_elements))
            
            encoded_element = tokenizer_instruction.encode_plus(
                    content["instruction"], ui_element["text"], padding="max_length", max_length=largest, truncation=True
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

