import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaPair
from layout_ipa.tasks.layout_lm.data_prep import PrepareLayoutLMPairTask
from layout_ipa.tasks.layout_lm.model_pipeline import LayoutLMPair
from sklearn.metrics import f1_score
from layout_ipa.util.evaluation import pair_evaluation

prepare_rico_task = PrepareRicoScaPair()

layout_lm_model = settings["layout_lm_base"]

train_path = settings["rico_sca"]["train"]
dev_path = settings["rico_sca"]["dev"]
test_path = settings["rico_sca"]["test"]

# train_path = settings["sample_rico_sca"]
# dev_path = settings["sample_rico_sca"]
# test_path = settings["sample_rico_sca"]

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"./cache/datasets/rico/"),
)

prepare_rico_task = PrepareRicoScaPair()
prepare_rico_layout_lm_task = PrepareLayoutLMPairTask()
layout_lm_trainer_task = LayoutLMPair()


with Flow("Running the Transformers for Pair Classification") as flow1:
    with tags("train"):
        train_input = prepare_rico_task(train_path)
        train_dataset = prepare_rico_layout_lm_task(train_input["data"])
    with tags("dev"):
        dev_input = prepare_rico_task(dev_path)
        dev_dataset = prepare_rico_layout_lm_task(dev_input["data"])
    with tags("test"):
        test_input = prepare_rico_task(test_path)
        test_dataset = prepare_rico_layout_lm_task(test_input["data"])
    layout_lm_trainer_task(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        mapping_dev=dev_input["mapping"],
        mapping_test=test_input["mapping"],
        task_name="layout_lm_pair_rico",
        output_dir="./cache/layout_lm_pair_rico/",
        eval_fn=pair_evaluation,
    )


FlowRunner(flow=flow1).run()
