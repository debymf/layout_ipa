from prefect import Task
from loguru import logger
from dynaconf import settings
import json


class PrepareRicoSca(Task):
    def run(self, file_location):
        parsed_data = dict()
        logger.info("Running Sample Task")
        with open(file_location, "r") as f:
            input_data = json.load(f)

        number_of_screens = len(input_data)
        total_pairs = 0
        total_negative_pairs = 0
        total_positive_pairs = 0
        for id_input, screen_info in input_data.items():
            ui_elements_dict = dict()
            index_ui_element = 0
            for ui_element in screen_info["ui_obj_str_seq"]:
                ui_elements_dict[index_ui_element] = {
                    "text": ui_element,
                    "x0": screen_info["ui_obj_cord_x_seq"][index_ui_element * 2],
                    "x1": screen_info["ui_obj_cord_x_seq"][(2 * index_ui_element) + 1],
                    "y0": screen_info["ui_obj_cord_y_seq"][index_ui_element * 2],
                    "y1": screen_info["ui_obj_cord_y_seq"][(2 * index_ui_element) + 1],
                }
                index_ui_element = index_ui_element + 1

            index_instruction = 0
            for instruction in screen_info["instruction_str"]:
                selected_ui_element = screen_info["ui_target_id_seq"][index_instruction]
                for ui_index, ui_element in ui_elements_dict.items():

                    if ui_index == selected_ui_element:
                        label_ui = 1
                        total_positive_pairs = total_positive_pairs + 1
                    else:
                        total_negative_pairs = total_negative_pairs + 1
                        label_ui = 0

                    parsed_data[total_pairs] = {
                        "instruction": instruction,
                        "ui": ui_element,
                        "label": label_ui,
                    }
                    total_pairs = total_pairs + 1

                index_instruction = index_instruction + 1

        logger.info(f"Number of different screens: {number_of_screens}.")
        logger.info(f"Total of pairs: {total_pairs}")
        logger.info(f"Total negative pairs: {total_negative_pairs}")
        logger.info(f"Total positive pairs: {total_positive_pairs}")

        return input_data
