from prefect import Task
from loguru import logger
from dynaconf import settings
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer


tokenizer_model = settings["layout_lm_base"]


class PrepareRicoLayoutLM(Task):
    def run(self, input_data, largest):
        logger.info("*** Preprocessing Data for LayoutLM ***")
        tokenizer = tokenizer_class.from_pretrained(tokenizer_model, do_lower_case=true)
        entries = dict()

        print(largest)
        for id_d, content in tqdm(data_dict.items()):
            encoded_dict = tokenizer.encode_plus(
                content["statement"],
                content["expression"],
                max_length=largest,
                pad_to_max_length=True,
            )

            start_span = content["span_start"]
            end_span = content["span_end"]
            if start_span == -1:
                start_span = largest
                end_span = largest
            entries[id_d] = {
                "statement": encoded_dict["input_ids"],
                "types": encoded_dict["token_type_ids"],
                "att": encoded_dict["attention_mask"],
                "start": start_span,
                "end": end_span,
            }
        result = pd.DataFrame.from_dict(entries, "index")
        return result