import prefect
from dynaconf import settings
from loguru import logger
from prefect import Flow, tags
from prefect.engine.flow_runner import FlowRunner
from prefect.engine.results import LocalResult
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaPair
from layout_ipa.tasks.transformers_based.data_prep import PrepareTransformersPairTask
from layout_ipa.tasks.transformers_based.model_pipeline import TransformerPair
from sklearn.metrics import f1_score

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
prepare_rico_transformer_task = PrepareTransformersPairTask()
transformer_trainer_task = TransformerPair()


with Flow("Running the Transformers for Pair Classification") as flow1:
    with tags("train"):
        train_input = prepare_rico_task(train_path)
        train_dataset = prepare_rico_transformer_task(train_input)
    with tags("dev"):
        dev_input = prepare_rico_task(dev_path)
        dev_dataset = prepare_rico_transformer_task(dev_input)
    with tags("test"):
        test_input = prepare_rico_task(test_path)
        test_dataset = prepare_rico_transformer_task(test_input)
    transformer_trainer_task(
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        test_dataset=test_dataset,
        task_name="transformer_pair_rico",
        output_dir="./cache/transformer_pair_rico/",
        eval_fn=f1_score,
    )


FlowRunner(flow=flow1).run()
