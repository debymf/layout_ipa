import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags, task
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaSelection
from layout_ipa.tasks.layout_lm.data_prep import PrepareLayoutLMSelectTask
from layout_ipa.tasks.layout_lm.model_pipeline import LayoutLMSelect
from sklearn.metrics import f1_score, accuracy_score
from layout_ipa.util.evaluation import pair_evaluation


layout_lm_model = settings["layout_lm_base"]

train_path = settings["rico_sca_sample"]["train"]
dev_path = settings["rico_sca_sample"]["dev"]
test_path = settings["rico_sca_sample"]["test"]

# train_path = settings["sample_rico_sca"]
# dev_path = settings["sample_rico_sca"]
# test_path = settings["sample_rico_sca"]

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"./cache/datasets/rico/"),
)

prepare_rico_task = PrepareRicoScaSelection()
prepare_rico_layout_lm_task = PrepareLayoutLMSelectTask()
layout_lm_trainer_task = LayoutLMSelect()

INSTRUCTION_TYPE = [0, 1, 2, 3]
#  where: 0 and 3 - Lexical Matching
#             1 - Spatial (Relative to screen)
#             2 - Spatial (Relative to other elements)


@task
def get_largest(train, dev, test):

    return max([train, dev, test])


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

    largest = get_largest(
        train_input["largest"], dev_input["largest"], test_input["largest"]
    )

    layout_lm_trainer_task(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        bert_model="microsoft/layoutlm-base-uncased",
        task_name="layout_lm_select_rico",
        output_dir="./cache/layout_lm_select_rico/",
        mode="train",
        eval_fn=accuracy_score,
        num_labels=largest,
    )


FlowRunner(flow=flow1).run()
