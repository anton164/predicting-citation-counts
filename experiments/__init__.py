from .base import Experiment
from .test import TestExperiment
from .magrank import MagRankExperiment

experiment_map = {"Test": TestExperiment, "Predicting MagRank": MagRankExperiment}
