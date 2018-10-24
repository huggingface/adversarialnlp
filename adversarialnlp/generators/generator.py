import logging
from typing import Dict, Union, Iterator, List

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

AdversarialExample = Union[str, Dict[str, Union[str, List[str]]]]  # pylint: disable=invalid-name


class Generator():
    """
    An abstract ``Generator`` class.
    Parameters
    ----------
    num_examples : int, optional (default = 8)
        Number of adversarial examples to yielded each time the iterator is called.
    """
    default_implementation = 'swag'

    def __init__(self,
                 seeds: Iterator,
                 num_examples: int = 8) -> None:
        self._num_examples = num_examples
        self._seeds = seeds

    def __iter__(self) -> Iterator[AdversarialExample]:
        """
        Returns a generator that yields batches of num_examples adversarial examples generated from
        the given dataset.
        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset to be used as seeds for generating the adversarial examples.
            IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        """
        raise NotImplementedError
