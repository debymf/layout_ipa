import argparse

import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.preprocessing.rico_sca import PrepareRicoSca, PrepareRicoLayoutLM
from layout_ipa.tasks.models import SeqTransformerTrainer


layout_lm_model = settings["layout_lm_base"]

train_path = settings["sample_rico_sca"]
dev_path = settings["sample_rico_sca"]
test_path = settings["sample_rico_sca"]

cache_args = dict(
    target="{task_name}-{task_tags}.pkl",
    checkpoint=True,
    result=LocalResult(dir=f"./cache/datasets/rico/"),
)
prepare_rico_task = PrepareRicoSca()
prepare_rico_layout_task = PrepareRicoLayoutLM()
transformer_trainer_task = SeqTransformerTrainer()


with Flow("Running the task with the RicoSCA dataset") as flow1:
    with tags("train"):
        train_input = prepare_rico_task(train_path)
        train_dataset = prepare_rico_layout_task(train_input)
    with tags("dev"):
        dev_input = prepare_rico_task(dev_path)
        dev_dataset = prepare_rico_layout_task(dev_input)
    with tags("test"):
        test_input = prepare_rico_task(test_path)
        test_dataset = prepare_rico_layout_task(test_input)
    # transformer_trainer_task(
    #     train_dataset=train_span_dataset,
    #     dev_dataset=dev_span_dataset,
    #     test_dataset=test_span_dataset,
    #     task_name="span_vtds",
    #     output_dir="./cache/span_vtds/",
    #     bert_model=BERT_MODEL,
    #     eval_fn=PrepareSpanTorchDataset.calculate_metrics,
    # )


FlowRunner(flow=flow1).run()
