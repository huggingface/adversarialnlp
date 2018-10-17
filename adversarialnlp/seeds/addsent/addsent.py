from adversarialnlp.generators import Generator
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

class AddSent(Generator):
    """Implement the AddSent method
    This class implements Robin Jia and Percy Liang's AddSent method for generating adversarial
    examples as described in `Adversarial Examples for Evaluating Reading Comprehension Systems
    <http://arxiv.org/abs/1707.07328>`_

    Parameters
    ----------
    method : ``str``, optional (default="AddSent")
        Currently accept the following methods:
            - AddSent

    """
    def __init__(self, dataset_reader: DatasetReader) -> None:
        super(AddSent, self).__init__(dataset_reader)
