import logging
from typing import Dict, Union, Iterable, Iterator, List, Optional, Tuple
from collections import defaultdict
import itertools
import math
import random

import torch

from allennlp.common.registrable import Registrable
from allennlp.common.util import is_lazy, lazy_groups_of, ensure_list
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators import DataIterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]  # pylint: disable=invalid-name


class Generator(Registrable):
    """
    An abstract ``Generator`` class. ``Generators`` must override ``_create_batches()``.
    Parameters
    ----------
    num_examples : int, optional (default = 8)
        Number of adversarial examples to yielded each time the iterator is called.
    """
    default_implementation = 'swag'

    def __init__(self,
                 num_examples: int = 8) -> None:
        self._num_examples = num_examples

    def __call__(self,
                 instances: DataIterator) -> Iterator[TensorDict]:
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
