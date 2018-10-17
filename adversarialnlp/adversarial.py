from typing import List

from allennlp.common import Registrable
from allennlp.models.model import Model

from adversarialnlp import Generator, Pruner

class Adversarial(Registrable):
    """
    ``Adversarial`` handles the general process of crafting adversarial NLP:
        a) Sampling from a dataset
        b) Edit samples with rules / random modification (can be constrained) / conditional generation
        c) Prune the potential AE using scores from a tested model / a language model / another type of model
        d) Iterating (genetic alg...)

    Parameters
    ----------
    dataset_reader : ``DatasetReader``
        The ``DatasetReader`` object that will be used to sample training examples.

    """
    def __init__(self,
                 generator: Generator,
                 pruner: Pruner,
                 num_samples: int = 10) -> None:
        self._generator = generator
        self._pruner = pruner
        self._num_samples = num_samples

    def forward(self, num_samples: int = -1):
        if num_samples == -1:
            num_samples = self._num_samples
        potential_examples = self._generator()
        pruned_examples = self._pruner(potential_examples, num_samples)
        return pruned_examples
