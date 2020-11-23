from .base import Experiment
from .test import TestExperiment
from .magrank import MagRankExperiment
from .journal import JournalExperiment

experiment_map = {
    "Test": TestExperiment,
    "Predicting MagRank": MagRankExperiment,
    "Predicting Journal": JournalExperiment,
}
