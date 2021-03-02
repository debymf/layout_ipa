import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags, task
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import (
    PrepareRicoScaSelection,
    PrepareRicoScaRegion,
)
from layout_ipa.tasks.ipa.data_prep import PrepareRegionLayoutLMTask
from layout_ipa.tasks.ipa.model_pipeline import LayoutLMRegionTrainer
from sklearn.metrics import accuracy_score
from layout_ipa.util.evaluation import pair_evaluation_vector
import os
import argparse

parser = argparse.ArgumentParser(description="Running flow for LayoutLM and Bert.")

parser.add_argument(
    "--type",
    metavar="Type of instruction",
    type=int,
    help="Type of instruction",
    default=[0, 1, 2, 3],
    nargs="+",
)


parser.add_argument(
    "--output_file",
    metavar="name of the output file",
    type=str,
    help="Output file",
    default="out.txt",
    nargs="?",
)

args = parser.parse_args()
INSTRUCTION_TYPE = args.type
FILENAME_RESULTS = args.output_file
#  where: 0 and 3 - Lexical Matching
#             1 - Spatial (Relative to screen)
#             2 - Spatial (Relative to other elements)

# train_path = settings["rico_sca"]["train"]
# dev_path = settings["rico_sca"]["dev"]
# test_path = settings["rico_sca"]["test"]

train_path = settings["sample_rico_sca"]
dev_path = settings["sample_rico_sca"]
test_path = settings["sample_rico_sca"]

# train_path = settings["rico_sca_sample"]["train"]
# dev_path = settings["rico_sca_sample"]["dev"]
# test_path = settings["rico_sca_sample"]["test"]

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"./cache/datasets/rico/"),
)

prepare_rico_selection_task = PrepareRicoScaSelection(**cache_args)
prepare_rico_region_task = PrepareRicoScaRegion(**cache_args)
prepare_rico_layout_lm_task = PrepareRegionLayoutLMTask(**cache_args)
layout_lm_trainer_task = LayoutLMRegionTrainer()

logger.success(f"***** TYPE {INSTRUCTION_TYPE} *****")
logger.success(f"***** OUTPUT FILE {FILENAME_RESULTS} *****")


@task
def save_output_results(output):
    if os.path.exists(FILENAME_RESULTS):
        append_write = "a"  # append if already exists
    else:
        append_write = "w"  # make a new file if not

    with open("./results/" + FILENAME_RESULTS, append_write) as f:
        f.write(f"TYPE: {INSTRUCTION_TYPE} \n")
        f.write(f"ACC DEV: {output['dev']['score']} \n")
        f.write(f"ACC TEST: {output['test']['score']} \n")
        f.write("=========================== \n \n")

    logger.info(f"TYPE: {INSTRUCTION_TYPE} \n")
    logger.info(f"ACC DEV: {output['dev']['score']} \n")
    logger.info(f"ACC TEST: {output['test']['score']} \n")
    logger.info("=========================== \n \n")


with Flow("Running flow for Bert and LayouLM") as flow1:
    with tags("train"):
        train_input = prepare_rico_selection_task(
            train_path, type_instructions=INSTRUCTION_TYPE
        )
        parsed_train = prepare_rico_region_task(train_input["data"])
        train_dataset = prepare_rico_layout_lm_task(parsed_train)
    with tags("dev"):
        dev_input = prepare_rico_selection_task(
            dev_path, type_instructions=INSTRUCTION_TYPE
        )
        parsed_dev = prepare_rico_region_task(dev_input["data"])
        dev_dataset = prepare_rico_layout_lm_task(parsed_dev)
    with tags("test"):
        test_input = prepare_rico_selection_task(
            test_path, type_instructions=INSTRUCTION_TYPE
        )
        parsed_test = prepare_rico_region_task(test_input["data"])
        test_dataset = prepare_rico_layout_lm_task(parsed_test)
    outputs = layout_lm_trainer_task(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        task_name="layout_lm_and_bert",
        output_dir="./cache/layout_lm_and_bert/",
        mode="train",
        eval_fn=accuracy_score,
    )
    # save_output_results(outputs)


FlowRunner(flow=flow1).run()
