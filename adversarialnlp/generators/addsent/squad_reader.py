import json
import logging
from typing import Iterator, List, Tuple

from adversarialnlp.common.file_utils import download_files

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def squad_reader(file_path: str = None) -> Iterator[List[Tuple[str, str]]]:
    r""" Reads a JSON-formatted SQuAD file and returns an Iterator.

    Args:
        file_path: Path to a JSON-formatted SQuAD file.
            If no path is provided, download and use SQuAD v1.0 training dataset.

    Return:
        list of tuple (question_answer, paragraph).
    """
    if file_path is None:
        file_path = download_files(fnames=['train-v1.1.json'],
                                   paths='https://rajpurkar.github.io/SQuAD-explorer/dataset/',
                                   local_folder='squad')
        file_path = file_path[0]

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
                out_data.append((question_answer, paragraph))
    return out_data
