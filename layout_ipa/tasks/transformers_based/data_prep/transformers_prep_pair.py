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


class PrepareTransformersPairTask(Task):
    def run(self, input_data, largest=256):
        logger.info("*** Preprocessing Data for Transformer-Based ***")
        tokenizer_instruction = BertTokenizer.from_pretrained("bert-base-uncased")
        entries = dict()

        for id_d, content in tqdm(input_data.items()):
            ui_element_text = content["ui"]["text"]

            encoded_element = tokenizer_instruction(
                content["instruction"],
                ui_element_text,
                padding="max_length",
                max_length=largest,
                truncation=True,
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

