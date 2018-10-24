import json
import logging
from typing import Iterator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def squad_reader(file_path: str) -> Iterator:
    """
    Reads a JSON-formatted SQuAD file and returns an Iterator over ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    file_path : ``str``
        Path to a JSON-formatted SQuAD file.
    """
    logger.info("Reading file at %s", file_path)
    with open(file_path) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    logger.info("Reading the dataset")
    out_data = []
    for article in dataset:
        for paragraph_json in article['paragraphs']:
            paragraph = paragraph_json["context"]
            for question_answer in paragraph_json['qas']:
                question_answer["question"] = question_answer["question"].strip().replace("\n", "")
                out_data.append((question_answer, paragraph, article['title']))
    return out_data
