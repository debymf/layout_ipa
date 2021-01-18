from prefect import Task
from loguru import logger
from dynaconf import settings
import json
import random

NUM_CHOICES = 20
SHUFFLE = False


class PrepareRicoScaSelect(Task):
    def shuffle_dict(self, entries):
        new_dict = dict()

        keys = list(entries.keys())
        random.shuffle(keys)

        mapping = dict()
        for key in keys:
            index_dict = len(new_dict)
            new_dict[index_dict] = entries[key]
            mapping[key] = index_dict

        # print("*** DICT ***")
        # print(new_dict)

        # print("*** NEW DICT ***")
        # print(entries)

        # print("*** MAPPING ***")
        # print(mapping)
        # input()

        return new_dict, mapping

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
        total_screen_elements = 0
        total_entries = 0
        largest = 0
        removed_entry = 0

        for _, screen_info in input_data.items():
            ui_elements_dict = dict()
            index_ui_element = 0
            if len(screen_info["ui_obj_str_seq"]) > NUM_CHOICES:
                removed_entry = removed_entry + 1
                continue

            total_screen_elements = (
                len(screen_info["ui_obj_str_seq"]) + total_screen_elements
            )
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
                index_ui_element = index_ui_element + 1
            # print("*** BEFORE ***")
            # print(ui_elements_dict)
            # input()
            if SHUFFLE:
                ui_elements_dict, mapping_ui_elements = self.shuffle_dict(
                    ui_elements_dict
                )

                # print("*** AFTER ***")
                # print(ui_elements_dict)
                # print(mapping_ui_elements)
                # input()
            index_instruction = 0
            for instruction in screen_info["instruction_str"]:
                if SHUFFLE:
                    selected_ui_element = mapping_ui_elements[
                        screen_info["ui_target_id_seq"][index_instruction]
                    ]
                else:
                    selected_ui_element = screen_info["ui_target_id_seq"][
                        index_instruction
                    ]
                if True:
                    # if screen_info["instruction_rule_id"][index_instruction] == 0 or screen_info["instruction_rule_id"][index_instruction] == 3:
                    # if screen_info["instruction_rule_id"][index_instruction] == 3:
                    parsed_data[total_entries] = {
                        "instruction": instruction,
                        "ui": ui_elements_dict,
                        "label": selected_ui_element,
                    }
                    # print(parsed_data[total_entries])

                    if len(screen_info["ui_obj_str_seq"]) > largest:
                        largest = len(screen_info["ui_obj_str_seq"])

                    total_entries = total_entries + 1

                index_instruction = index_instruction + 1

        logger.info(f"Largest index of selected UI element:{largest}")
        logger.info(f"Number of different screens: {number_of_screens}.")
        logger.info(f"Total Entries: {total_entries}")
        logger.info(f"Number of removed entries: {removed_entry}")

        return parsed_data
