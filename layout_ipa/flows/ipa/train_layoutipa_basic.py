# python -m  layout_ipa.flows.ipa.train_layoutipa_simple --weight_decay==0.0001 --learning_rate=0.00001 --type_screen_agg=0  --type_end_combine=0 --output_file "out1.txt"

import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags, task
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaScreenPair
from layout_ipa.tasks.ipa.data_prep import PrepareLayoutIpaBasic
from layout_ipa.tasks.ipa.model_pipeline import LayoutIpaBasicTrainer
from sklearn.metrics import f1_score
from layout_ipa.util.evaluation import pair_evaluation
import os
import argparse

parser = argparse.ArgumentParser(description="Running flow for Layout IPA.")

parser.add_argument(
    "--type",
    metavar="Type of instruction",
    type=int,
    help="Type of instruction",
    default=[2],
    nargs="+",
)

parser.add_argument(
    "--type_screen_agg",
    metavar="",
    type=int,
    help="0 - Deepset + FC; 1- FC; 2- Average; 3- Sum",
    default=0,
    nargs="?",
)

parser.add_argument(
    "--type_end_combine",
    metavar="",
    type=int,
    help="0 - Matching; 1 - Concat; 2- Sum; 3- Mult",
    default=0,
    nargs="?",
)


parser.add_argument(
    "--learning_rate",
    metavar="Learning rate",
    type=float,
    help="Learning rate",
    default=0.00001,
    nargs="?",
)


parser.add_argument(
    "--dropout", metavar="Dropout", type=float, help="Dropout", default=0.7, nargs="?",
)

parser.add_argument(
    "--weight_decay",
    metavar="Weight decay",
    type=float,
    help="weight decay",
    default=0.001,
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
#  where: 0 and 3 - Lexical Matching
#             1 - Spatial (Relative to screen)
#             2 - Spatial (Relative to other elements)
FILENAME_RESULTS = args.output_file
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
SCREEN_AGG = args.type_screen_agg
COMBINE_OUTPUT = args.type_end_combine
DROPOUT = args.dropout


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

prepare_rico_task = PrepareRicoScaScreenPair()
prepare_rico_layout_lm_task = PrepareLayoutIpaBasic()

bert_param = {
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
}


layout_lm_trainer_task = LayoutIpaBasicTrainer(**bert_param)

logger.success(f"*********************************************")
logger.success(f"***** OUTPUT FILE {FILENAME_RESULTS} *****")
logger.success(f"SCREEN AGG: {SCREEN_AGG}")
logger.success(f"OUTPUT COMBINE: {COMBINE_OUTPUT}")
logger.success(f"LEARNING RATE: {LEARNING_RATE}")
logger.success(f"WEIGHT DECAY: {WEIGHT_DECAY}")
logger.success(f"TYPE: {INSTRUCTION_TYPE}")
logger.success(f"DROPOUT: {DROPOUT}")
logger.success(f"*********************************************")


@task
def save_output_results(output):
    if os.path.exists("./results/" + FILENAME_RESULTS):
        append_write = "a"  # append if already exists
    else:
        append_write = "w"  # make a new file if not

    with open("./results/" + FILENAME_RESULTS, append_write) as f:
        f.write(f"TYPE: {INSTRUCTION_TYPE} \n")
        f.write(f"LEARNING RATE: {LEARNING_RATE} \n")
        f.write(f"WEIGHT DECAY: {WEIGHT_DECAY} \n")
        f.write(f"DROPOUT: {DROPOUT} \n")
        f.write(
            f"SCREEN AGG: {SCREEN_AGG} (0 - Deepset + FC; 1- FC; 2- Average; 3- Sum) \n"
        )
        f.write(
            f"OUTPUT COMBINE: {COMBINE_OUTPUT} (0 - Matching; 1 - Concat; 2- Sum; 3- Mult)\n"
        )
        f.write(f"ACC DEV: {output['dev']['score']} \n")
        f.write(f"ACC TEST: {output['test']['score']} \n")
        f.write("=========================== \n \n")

    logger.info(
        f"SCREEN AGG: {SCREEN_AGG} (0 - Deepset + FC; 1- FC; 2- Average; 3- Sum)\n"
    )
    logger.info(
        f"OUTPUT COMBINE: {COMBINE_OUTPUT} (0 - Matching; 1 - Concat; 2- Sum; 3- Mult)\n"
    )
    logger.info(f"DROPOUT: {DROPOUT} \n")
    logger.info(f"LEARNING RATE: {LEARNING_RATE} \n")
    logger.info(f"WEIGHT DECAY: {WEIGHT_DECAY} \n")
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
        task_name="layout_ipa_last_hope",
        output_dir="./cache/layout_ipa_last_hope/",
        mode="train",
        eval_fn=pair_evaluation,
        screen_arg=SCREEN_AGG,
        combine_output=COMBINE_OUTPUT,
        dropout=DROPOUT,
    )
    save_output_results(outputs)


FlowRunner(flow=flow1).run()
