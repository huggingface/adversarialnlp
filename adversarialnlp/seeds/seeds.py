"""
A :class:`~Seeds` represents a collection of ``Instance`` s to be fed
through an ``Editor``. It's a sub-class of ``Batch``.
"""

import logging
from typing import Iterable

from allennlp.data.instance import Instance
from allennlp.data.dataset import Batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Seeds(Batch):
    """
    A batch of Instances to be used in an Adversarial Editor.
	In addition to containing the instances themselves, it contains helper functions for converting
	the data into tensors.
    """
    def __init__(self, instances: Iterable[Instance]) -> None:
        """
        A Seed just takes an iterable of instances in its constructor and hangs onto them
        in a list.
        """
        super().__init__(instances)
