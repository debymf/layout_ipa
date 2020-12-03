import argparse

import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.preprocessing.rico_sca import (
    PrepareRicoScaSelect,
    PrepareRicoTransformerSelect,
)
from layout_ipa.tasks.models import SelectionTransformerTrainer
from sklearn.metrics import accuracy_score

layout_lm_model = settings["layout_lm_base"]

train_path = settings["rico_sca_sample"]["train"]
dev_path = settings["rico_sca_sample"]["dev"]
test_path = settings["rico_sca_sample"]["test"]

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"./cache/datasets/rico/"),
)
prepare_rico_task = PrepareRicoScaSelect(**cache_args)
prepare_rico_transformer_task = PrepareRicoTransformerSelect(**cache_args)
transformer_trainer_task = SelectionTransformerTrainer()


with Flow("Running the task with the RicoSCA dataset") as flow1:
    with tags("train"):
        train_input = prepare_rico_task(train_path)
        train_dataset = prepare_rico_transformer_task(train_input, bert_model="bert-base-uncased")
    with tags("dev"):
        dev_input = prepare_rico_task(train_path)
        dev_dataset = prepare_rico_transformer_task(train_input, bert_model="bert-base-uncased",)
    with tags("test"):
        test_input = prepare_rico_task(test_path)
        test_dataset = prepare_rico_transformer_task(test_input, bert_model="bert-base-uncased",)
    transformer_trainer_task(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        task_name="span_vtds",
        output_dir="./cache/span_vtds/",
        eval_fn=accuracy_score,
    )


FlowRunner(flow=flow1).run()
