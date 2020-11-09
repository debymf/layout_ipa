import unittest
from layout_ipa.tasks.preprocessing.rico_sca import PrepareRicoSca, PrepareRicoLayoutLM
from dynaconf import settings
from loguru import logger

FILE_LOCATION = settings["sample_rico_sca"]


class PrepareRicoScaTest(unittest.TestCase):
    def test_prep_rico(self):
        rico_prep_task = PrepareRicoSca()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        rico_layoutlm_prep = PrepareRicoLayoutLM()
        rico_layoutlm_prep.run(result_rico)
