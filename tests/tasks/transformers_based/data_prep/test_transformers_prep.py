import unittest
from layout_ipa.tasks.datasets_parse.rico_sca import (
    PrepareRicoScaPair,
    PrepareRicoScaSelection,
    PrepareRicoScaEmbedding,
)

from layout_ipa.tasks.transformers_based.data_prep import PrepareTransformersPairTask
from dynaconf import settings
from loguru import logger

FILE_LOCATION = settings["sample_rico_sca"]


class PrepareRicoScaTransformersTest(unittest.TestCase):
    def test_prep_rico_transformer_pair(self):
        rico_prep_task = PrepareRicoScaPair()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        rico_layoutlm_prep = PrepareTransformersPairTask()
        layout_prep_rico = rico_layoutlm_prep.run(result_rico)

        i = 0
        for content in layout_prep_rico:
            logger.info(f"Content: {content}")

            i = i + 1
            if i == 10:
                break

    def test_prep_rico_transformer_select(self):
        rico_prep_task = PrepareRicoScaPair()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        rico_layoutlm_prep = PrepareTransformersPairTask()
        layout_prep_rico = rico_layoutlm_prep.run(result_rico)

        i = 0
        for content in layout_prep_rico:
            logger.info(f"Content: {content}")

            i = i + 1
            if i == 10:
                break

    def test_prep_rico_transformer_embedding(self):
        rico_prep_task = PrepareRicoScaPair()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        rico_layoutlm_prep = PrepareTransformersPairTask()
        layout_prep_rico = rico_layoutlm_prep.run(result_rico)

        i = 0
        for content in layout_prep_rico:
            logger.info(f"Content: {content}")

            i = i + 1
            if i == 10:
                break
