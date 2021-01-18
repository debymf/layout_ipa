import unittest
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaPair
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaSelection
from layout_ipa.tasks.datasets_parse.rico_sca import PrepareRicoScaEmbedding
from dynaconf import settings
from loguru import logger

FILE_LOCATION = settings["sample_rico_sca"]


class PrepareRicoScaTest(unittest.TestCase):
    def test_prep_rico_pair(self):
        logger.info("**** TEST PAIR CLASSIFICATION *****")
        logger.info("All Types")
        rico_prep_task = PrepareRicoScaPair()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 0")
        result_rico = rico_prep_task.run(FILE_LOCATION, [0])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 0 and 3")
        result_rico = rico_prep_task.run(FILE_LOCATION, [0, 3])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 1")
        result_rico = rico_prep_task.run(FILE_LOCATION, [1])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 2")
        result_rico = rico_prep_task.run(FILE_LOCATION, [2])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("*****************************************")

    def test_prep_rico_selection(self):
        logger.info("**** TEST PREP SELECION *****")
        rico_prep_task = PrepareRicoScaSelection()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 0")
        result_rico = rico_prep_task.run(FILE_LOCATION, [0])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 0 and 3")
        result_rico = rico_prep_task.run(FILE_LOCATION, [0, 3])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 1")
        result_rico = rico_prep_task.run(FILE_LOCATION, [1])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 2")
        result_rico = rico_prep_task.run(FILE_LOCATION, [2])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("*****************************************")

    def test_prep_rico_embedding(self):
        logger.info("**** TEST PREP EMBEDDING *****")
        rico_prep_task = PrepareRicoScaEmbedding()
        result_rico = rico_prep_task.run(FILE_LOCATION)

        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 0")
        result_rico = rico_prep_task.run(FILE_LOCATION, [0])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 0 and 3")
        result_rico = rico_prep_task.run(FILE_LOCATION, [0, 3])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 1")
        result_rico = rico_prep_task.run(FILE_LOCATION, [1])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("Type 2")
        result_rico = rico_prep_task.run(FILE_LOCATION, [2])
        for id_r, elements in result_rico.items():
            print(id_r)
            print(elements)
            break

        logger.info("*****************************************")

