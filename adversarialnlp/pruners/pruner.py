from allennlp.common import Registrable

class Pruner(Registrable):
    """
    ``Pruner`` is used to fil potential adversarial samples

    Parameters
    ----------
    dataset_reader : ``DatasetReader``
        The ``DatasetReader`` object that will be used to sample training examples.

    """
    def __init__(self, ) -> None:
        super().__init__()
