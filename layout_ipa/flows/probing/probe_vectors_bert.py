import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags, task
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaScreenPair
from layout_ipa.tasks.probing import PrepareBertProbing
from layout_ipa.tasks.probing import GetVectorsBertProbing, AddExtraLabelsTask
from layout_ipa.tasks.ipa.data_prep import PrepareBertandLayoutLM
from layout_ipa.tasks.ipa.model_pipeline import BertandLayoutLMTrainer
from sklearn.metrics import f1_score
from layout_ipa.util.evaluation import pair_evaluation_vector
import os
from tqdm import tqdm
import argparse
import csv
import pandas as pd

#  where: 0 and 3 - Lexical Matching
#             1 - Spatial (Relative to screen)
#             2 - Spatial (Relative to other elements)

# MODEL_LOCATION = "/nobackup/projects/bdman04/layout_ipa/cache/transformer_pair_rico/transformer_pair_rico/"
MODEL_LOCATION = "bert-base-uncased"
# OUTPUT_METADATA = "./results/bert_vectors_meta_data.tsv"
OUTPUT = "./results/bert_vectors_new_labels.tsv"
# test_path = settings["rico_sca"]["test"]
test_path = settings["sample_rico_sca"]

prepare_rico_task = PrepareRicoScaScreenPair()
add_extra_labels = AddExtraLabelsTask()
prepare_data_for_probing = PrepareBertProbing()
get_vectors_task = GetVectorsBertProbing()


@task
def save_output(semantic, absolute, relative):

    logger.info(f"TOTAL semantic: {len(semantic['instruction'])}")
    logger.info(f"TOTAL absolute: {len(absolute['instruction'])}")
    logger.info(f"TOTAL relative: {len(relative['instruction'])}")
    output_dict = dict()
    output_dict["representation"] = list()
    output_dict["labels"] = list()
    output_dict["instruction"] = list()
    output_dict["ui_text"] = list()
    output_dict["type"] = list()
    output_dict["is_top"] = list()
    output_dict["is_right"] = list()
    dimensions_out = dict()

    for i in range(0, 768):
        output_dict[f"x{i}"] = list()

    for key, content in semantic.items():
        output_dict[key].extend(content)
    for key, content in absolute.items():
        output_dict[key].extend(content)
    for key, content in relative.items():
        output_dict[key].extend(content)

    for representation in tqdm(output_dict["representation"]):
        for i in range(0, len(representation)):
            output_dict[f"x{i}"].append(representation[i])
    output_dict.pop("representation", None)
    output_frame = pd.DataFrame.from_dict(output_dict)
    output_frame.to_csv(OUTPUT, sep="\t", index=False)


# New type semattic = 0 -> Semantic 1-> Absolute 2->Relative
with Flow("Running flow for Bert and LayouLM") as flow1:
    input_semantic = prepare_rico_task(test_path, type_instructions=[0, 3], limit=50)
    input_semantic_parsed = add_extra_labels(input_semantic["data"])
    dataset_semantic = prepare_data_for_probing(input_semantic_parsed, 0)

    input_spatial_absolute = prepare_rico_task(
        test_path, type_instructions=[1], limit=50
    )
    input_spatial_absolute_parsed = add_extra_labels(input_spatial_absolute["data"])
    dataset_spatial_absolute = prepare_data_for_probing(
        input_spatial_absolute_parsed, 1
    )

    input_spatial_relative = prepare_rico_task(
        test_path, type_instructions=[2], limit=50
    )
    input_spatial_relative_parsed = add_extra_labels(input_spatial_relative["data"])
    dataset_spatial_relative = prepare_data_for_probing(
        input_spatial_relative_parsed, 2
    )

    output_semantic = get_vectors_task(
        dataset=dataset_semantic, model_location=MODEL_LOCATION
    )
    output_spatial_absolute = get_vectors_task(
        dataset=dataset_spatial_absolute, model_location=MODEL_LOCATION
    )
    output_spatial_relative = get_vectors_task(
        dataset=dataset_spatial_relative, model_location=MODEL_LOCATION
    )
    save_output(output_semantic, output_spatial_absolute, output_spatial_relative)


FlowRunner(flow=flow1).run()
