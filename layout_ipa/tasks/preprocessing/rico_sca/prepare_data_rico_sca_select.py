from prefect import Task
from loguru import logger
from dynaconf import settings
import json


class PrepareRicoScaSelect(Task):
    def run(self, file_location):
        """Parses the RicoSCA dataset.

        Args:
            file_location (str): location of the RicoSCA dataset

        Returns:
            Dict: preprocessed dict in the following format:
            
            {instruction: NL intruction,
            ui: DICT:
                    text: text of the ui element,
                    x0: bounding box x0,
                    x1: bouding box x1,
                    y0: bounding box y0,
                    y1: bounding box y1,
            label: From the list of UI elements, obtain the one reffered in the sentence.
            }
        """

        parsed_data = dict()
        logger.info("Preprocessing Rico SCA dataset")
        with open(file_location, "r") as f:
            input_data = json.load(f)

        number_of_screens = len(input_data)
        total_entries = 0

        for _, screen_info in input_data.items():
            ui_elements_dict = dict()
            index_ui_element = 0

            for ui_element in screen_info["ui_obj_str_seq"]:
                ui_elements_dict[index_ui_element] = {
                    "text": ui_element,
                    "x0": screen_info["ui_obj_cord_x_seq"][index_ui_element * 2] * 1000,
                    "x1": screen_info["ui_obj_cord_x_seq"][(2 * index_ui_element) + 1]
                    * 1000,
                    "y0": screen_info["ui_obj_cord_y_seq"][index_ui_element * 2] * 1000,
                    "y1": screen_info["ui_obj_cord_y_seq"][(2 * index_ui_element) + 1]
                    * 1000,
                }
                # print(ui_elements_dict[index_ui_element])
                # print(
                #     f"height: {ui_elements_dict[index_ui_element]['y1']-ui_elements_dict[index_ui_element]['y0']}"
                # )

                # print(
                #     f"weight: {ui_elements_dict[index_ui_element]['x1']-ui_elements_dict[index_ui_element]['x0']}"
                # )

                # input()
                index_ui_element = index_ui_element + 1

            index_instruction = 0
            for instruction in screen_info["instruction_str"]:
                selected_ui_element = screen_info["ui_target_id_seq"][index_instruction]
                parsed_data[total_entries] = {
                    "instruction": instruction,
                    "ui": ui_elements_dict,
                    "label": selected_ui_element,
                }
                total_entries = total_entries + 1

                index_instruction = index_instruction + 1

        logger.info(f"Number of different screens: {number_of_screens}.")
        logger.info(f"Total Entries: {total_entries}")

        return parsed_data
