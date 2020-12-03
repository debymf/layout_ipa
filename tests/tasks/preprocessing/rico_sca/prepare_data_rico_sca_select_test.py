import unittest
from layout_ipa.tasks.preprocessing.rico_sca import (
    PrepareRicoScaSelect,
    PrepareRicoLayoutLMSelect,
    PrepareRicoTransformerSelect,
)
from dynaconf import settings
from loguru import logger

FILE_LOCATION = settings["sample_rico_sca"]


class PrepareRicoScaSelectTest(unittest.TestCase):
    # def test_prep_rico(self):
    #     rico_prep_task = PrepareRicoScaSelect()
    #     result_rico = rico_prep_task.run(FILE_LOCATION)

    #     rico_layoutlm_prep = PrepareRicoLayoutLMSelect()
    #     layout_prep_rico = rico_layoutlm_prep.run(result_rico)

    #     for content in layout_prep_rico:
    #         logger.info(f"Content: {content}")

    def test_prep_rico_transformer(self):
        rico_prep_task = PrepareRicoScaSelect()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        rico_layoutlm_prep = PrepareRicoTransformerSelect()
        layout_prep_rico = rico_layoutlm_prep.run(result_rico, "bert-base-uncased")

        for content in layout_prep_rico:
            logger.info(f"Content: {content}")
