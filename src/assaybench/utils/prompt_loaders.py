import yaml
import os
from importlib.resources import files
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_objective_prompt(keyword, think_open_tag = None, think_close_tag = None, answer_open_tag= None, answer_close_tag= None, file_name: str = "objective_prompts.yaml"):
    """
    Loads the objective prompt of the keyword keyword
    """
    yaml_path = files("assaybench.data.prompts")/file_name
    with open(yaml_path, "r") as f:
        prompts = yaml.safe_load(f)

    prompt = prompts.get(keyword, "")
    if prompt == "":
        log.warning(f"Objective prompt with keyword {keyword} was not found. returned empty string instead")

    if ("think_open_tag" in prompt) and (think_open_tag is None):
        log.warning(f"Prompt contains think_open_tag but it was not provided.")
    if ("think_close_tag" in prompt) and (think_close_tag is None):
        log.warning(f"Prompt contains think_close_tag but it was not provided.")
    if ("answer_open_tag" in prompt) and (answer_open_tag is None):
        log.warning(f"Prompt contains answer_open_tag but it was not provided.")
    if ("answer_close_tag" in prompt) and (answer_close_tag is None):
        log.warning(f"Prompt contains answer_close_tag but it was not provided.")

    prompt = prompt.format(think_open_tag = think_open_tag,
                           think_close_tag = think_close_tag,
                           answer_open_tag = answer_open_tag,
                           answer_close_tag = answer_close_tag)

    return prompt
