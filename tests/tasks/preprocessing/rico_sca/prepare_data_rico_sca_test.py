import unittest
from layout_ipa.tasks.preprocessing.rico_sca import PrepareRicoSca
from dynaconf import settings


FILE_LOCATION = settings["sample_rico_sca"]


class PrepareRicoScaTest(unittest.TestCase):
    def test_prep_rico(self):
        rico_prep_task = PrepareRicoSca()
        rico_prep_task.run(FILE_LOCATION)
