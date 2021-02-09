import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags, task
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaPair
from layout_ipa.tasks.ipa.data_prep import PrepareLayoutIpaSimple
from layout_ipa.tasks.ipa.model_pipeline import LayoutIpaSimpleTrainer
from sklearn.metrics import f1_score
from layout_ipa.util.evaluation import pair_evaluation
import os
import argparse

parser = argparse.ArgumentParser(description="Running flow for Layout IPA.")

parser.add_argument(
    "--type",
    metavar="Type of instruction",
    type=list,
    help="Type of instruction",
    default=[0, 1, 2, 3],
    nargs="?",
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

# train_path = settings["sample_rico_sca"]
# dev_path = settings["sample_rico_sca"]
# test_path = settings["sample_rico_sca"]

train_path = settings["rico_sca_sample"]["train"]
dev_path = settings["rico_sca_sample"]["dev"]
test_path = settings["rico_sca_sample"]["test"]

# cache_args = dict(
#     target="{task_name}-{task_tags}.pkl",
#     checkpoint=True,
#     result=LocalResult(dir=f"./cache/datasets/rico/"),
# )

prepare_rico_task = PrepareRicoScaPair()
prepare_rico_layout_lm_task = PrepareLayoutIpaSimple()
layout_lm_trainer_task = LayoutIpaSimpleTrainer()

logger.succes(f"***** TYPE {INSTRUCTION_TYPE} *****")


@task
def save_output_results(output):
    if os.path.exists(FILENAME_RESULTS):
        append_write = "a"  # append if already exists
    else:
        append_write = "w"  # make a new file if not

    with open(FILENAME_RESULTS, append_write) as f:
        f.write(f"TYPE: {INSTRUCTION_TYPE} \n")
        f.write(f"ACC DEV: {output['dev']['score']} \n")
        f.write(f"ACC TEST: {output['test']['score']} \n")
        f.write("=========================== \n \n")

    logger.info(f"TYPE: {INSTRUCTION_TYPE} \n")
    logger.info(f"ACC DEV: {output['dev']['score']} \n")
    logger.info(f"ACC TEST: {output['test']['score']} \n")
    logger.info("=========================== \n \n")


with Flow("Running the Transformers for Pair Classification") as flow1:
    with tags("train"):
        train_input = prepare_rico_task(train_path, type_instructions=INSTRUCTION_TYPE)
        train_dataset = prepare_rico_layout_lm_task(train_input["data"])
    with tags("dev"):
        dev_input = prepare_rico_task(dev_path, type_instructions=INSTRUCTION_TYPE)
        dev_dataset = prepare_rico_layout_lm_task(dev_input["data"])
    with tags("test"):
        test_input = prepare_rico_task(test_path, type_instructions=INSTRUCTION_TYPE)
        test_dataset = prepare_rico_layout_lm_task(test_input["data"])
    outputs = layout_lm_trainer_task(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        mapping_dev=dev_input["mapping"],
        mapping_test=test_input["mapping"],
        task_name="layout_ipa_simple_pair_rico",
        output_dir="./cache/layout_ipa_simple_pair_rico/",
        mode="train",
        eval_fn=pair_evaluation,
    )
    save_output_results(outputs)


FlowRunner(flow=flow1).run()
