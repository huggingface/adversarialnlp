from adversarialnlp import Adversarial
from allennlp.data.dataset_readers.reading_comprehension.squad import SquadReader

adversarial = Adversarial(dataset_reader=SquadReader, editor='lstm_lm', num_samples=10)
examples = adversarial.generate()