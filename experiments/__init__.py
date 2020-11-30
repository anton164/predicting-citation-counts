from .base import Experiment
from .test import TestExperiment
from .binned_citation_count import BinnedCitationCountExperiment
from .citation_count_regression import CitationCountRegressionExperiment
from .journal import JournalExperiment

experiment_map = {
    "Test": TestExperiment,
    "Citation Count Regression": CitationCountRegressionExperiment,
    "Predicting Binned Citation Count": BinnedCitationCountExperiment,
    "Predicting Journal": JournalExperiment,
}
