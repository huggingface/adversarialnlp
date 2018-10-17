from allennlp.common import Registrable
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

class SeedsGenerator(Registrable):
    """
    ``SeedsGenerator`` prepare ``Seeds`` to be used by an ``Editor`` to create an adversarial example.
    ``Seeds`` are created from examples sampled from the dataset.

    Parameters
    ----------
    dataset_reader : ``DatasetReader``
        The ``DatasetReader`` object that will be used to sample examples for preparing ``Seeds`` objects.

    """
    def __init__(self, dataset_reader: DatasetReader) -> None:
        super().__init__()
        self._dataset_reader = dataset_reader
