# Code adapted from https://github.com/microsoft/unilm/blob/master/layoutlm/layoutlm/data/funsd.py

from prefect import Task
from loguru import logger
from dynaconf import settings
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer
from torch.utils.data import Dataset
import torch

tokenizer_model = "microsoft/layoutlm-base-uncased"


class PrepareLayoutIpaSimple(Task):
    def run(
        self,
        input_data,
        bert_model="bert-base-uncased",
        largest=512,
        largest_instruction=5,
    ):
        logger.info("*** Preprocessing Data for Layout IPA (simple) ***")
        tokenizer_layout = AutoTokenizer.from_pretrained(tokenizer_model)
        tokenizer_instruction = BertTokenizer.from_pretrained(bert_model)
        entries = dict()
        for id_d, content in tqdm(input_data.items()):

            encoded_ui = self.convert_examples_to_features(
                content["instruction"], content["ui"], largest, tokenizer_layout,
            )

            # encoded_instruction = tokenizer_instruction.encode_plus(
            #     content["instruction"],
            #     content["ui"]["text"],
            #     padding="max_length",
            #     max_length=largest_instruction,
            # )

            entries[id_d] = {
                "id_query": content["id_query"],
                "ui_position": content["ui_position"],
                "inst_input_ids": encoded_instruction["input_ids"],
                "inst_att_mask": encoded_instruction["attention_mask"],
                "inst_token_ids": encoded_instruction["token_type_ids"],
                "ui_input_ids": encoded_ui["ui_input_ids"],
                "ui_att_mask": encoded_ui["ui_input_mask"],
                "ui_token_ids": encoded_ui["ui_segment_ids"],
                "ui_boxes": encoded_ui["ui_boxes"],
                "label": content["label"],
            }

        return TorchDataset(entries)

    @staticmethod
    def convert_examples_to_features(
        instruction,
        example,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
        pad_token_segment_id=0,
        pad_token_label_id=-1,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        features = dict()

        tokens = []
        token_boxes = []
        box = [
            int(example["x0"]),
            int(example["y0"]),
            int(example["x1"]),
            int(example["y1"]),
        ]
        word_tokens = tokenizer.tokenize(instruction + "[SEP]" + example["text"])
        # word_tokens = tokenizer.tokenize(example["text"])
        tokens.extend(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        token_boxes += [sep_token_box]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_boxes += [sep_token_box]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            token_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length

        features["ui_input_ids"] = input_ids
        features["ui_input_mask"] = input_mask
        features["ui_segment_ids"] = segment_ids
        features["ui_boxes"] = token_boxes

        return features


class TorchDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = list(dataset.values())
        self.keys = list(dataset.keys())

    def __getitem__(self, index):
        instance = self.dataset[index]
        return (
            torch.LongTensor(instance["inst_input_ids"]),
            torch.LongTensor(instance["inst_att_mask"]),
            torch.LongTensor(instance["inst_token_ids"]),
            torch.LongTensor([0]),
            torch.LongTensor([0]),
            torch.LongTensor([0]),
            torch.LongTensor([0]),
            instance["label"],
            instance["id_query"],
            instance["ui_position"],
            index,
        )
        #  return (
        #     torch.LongTensor(instance["inst_input_ids"]),
        #     torch.LongTensor(instance["inst_att_mask"]),
        #     torch.LongTensor(instance["inst_token_ids"]),
        #     torch.LongTensor(instance["ui_input_ids"]),
        #     torch.LongTensor(instance["ui_att_mask"]),
        #     torch.LongTensor(instance["ui_token_ids"]),
        #     torch.LongTensor(instance["ui_boxes"]),
        #     instance["label"],
        #     instance["id_query"],
        #     instance["ui_position"],
        #     index,
        # )

    def get_id(self, index):
        return self.keys[index]

    def __len__(self):
        return len(self.dataset)

