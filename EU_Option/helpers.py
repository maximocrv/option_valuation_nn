"""Define functions for generating model gradients, plots containing histograms and goodness of fit lines."""
import os
from pathlib import Path


def get_base_path():
    """
    Return a string containing the base path pertaining to the EU_Option folder. Set this environment variable by
    editing the run configurations and adding an environment variable.
    :return: String containing file path to the root folder of this project.
    """
    base_path = Path(os.environ['EU_Option'])
    return base_path


def listdir_nohidden(path):
    for ele in os.listdir(path):
        if not ele.startswith('.'):
            yield ele
