from .base import Experiment
from .test import TestExperiment
from .anton import AntonExperiment

experiment_map = {"Test": TestExperiment, "Anton": AntonExperiment}
