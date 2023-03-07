import yaml
import pandas as pd
import os
import numpy as np
import logging


def read_config(yaml_file):
    """Reads in the YAML config file."""
    with open(yaml_file, "r") as infile:
        yamldata = yaml.safe_load(infile)

    return yamldata


def add_file_label(filename, label):
    """Injects a model identifier to the outfile name."""
    splitfile = filename.split(".")
    newname = f"{''.join(splitfile[:-1])}_{label}.{splitfile[-1]}"
    return newname


def get_scenario_from_filename(filename, scenarios):
    for s in scenarios:
        if s in filename:
            return s


def get_rep_id(filepath):
    """Searches path for integer, uses as id for replicate."""
    for i in filepath.split("/"):
        try:
            int(i)
            return i
        except ValueError:
            continue


def get_logger(module_name):
    logging.basicConfig()
    logger = logging.getLogger(module_name)
    logger.setLevel("INFO")

    return logger
