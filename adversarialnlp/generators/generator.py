import logging
from typing import Dict, Union, Iterable, List
from collections import defaultdict
import itertools

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Generator():
    r"""An abstract ``Generator`` class.
    Parameters
    ----------
    num_examples : int, optional (default = 8)
        Number of adversarial examples to yielded each time the iterator is called.
    """
    default_implementation = 'addsent'

    def __init__(self,
                 default_seeds: Iterable = None,
                 quiet: bool = False):
        self.quiet: bool = quiet
        self._epochs: Dict[int, int] = defaultdict(int)
        self.default_seeds: Iterable = default_seeds

    def _generate_from_seed(self, seed: any):
        r"""Generate an adversarial example from a seed.
        """
        raise NotImplementedError

    def __call__(self,
                 seeds: Iterable = None,
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterable:
        r"""Generate adversarial examples using _generate_from_seed.

        Args:
            seeds: Instances to use as seed for adversarial
                example generation.
            num_epochs: How many times should we iterate over the seeds?
                If None, we will iterate over it forever.
            shuffle: Shuffle the instances before iteration.
                If True, we will shuffle the instances before iterating.

        Yields: adversarial_examples
            adversarial_examples: Adversarial examples generated
            from the seeds.
        """
        if seeds is None:
            if self.default_seeds is not None:
                seeds = self.default_seeds
            else:
                return
        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(seeds)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            self._epochs[key] = epoch
            for seed in seeds:
                yield from self._generate_from_seed(seed)
