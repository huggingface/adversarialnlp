from typing import Dict
import json
import logging
from overrides import overrides
from unidecode import unidecode

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

from adversarialnlp.generators.swag.utils import pairwise, postprocess

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("activitynet_captions")
class ActivityNetCaptionsDatasetReader(DatasetReader):
    r""" Reads ActivityNet Captions JSON files and creates a dataset suitable for crafting
    adversarial examples with swag using these captions.

    Expected format:
        JSON dict[video_id, video_obj] where
                video_id: str,
                video_obj:  {
                                "duration": float,
                                "timestamps": list of pairs of float,
                                "sentences": list of strings
                            }

    The output of ``read`` is a list of ``Instance`` s with the fields:
        video_id: ``MetadataField``
        first_sentence: ``TextField``
        second_sentence: ``TextField``

    The instances are created from all consecutive pair of sentences
    associated to each video.
    Ex: if a video has three associated sentences: s1, s2, s3 read will
    generate two instances:

        1. Instance("first_sentence" = s1, "second_sentence" = s2)
        2. Instance("first_sentence" = s2, "second_sentence" = s3)

    Args:
        lazy : If True, training will start sooner, but will take
            longer per batch. This  allows training with datasets that
            are too large to fit in memory. Passed to DatasetReader.
        tokenizer : Tokenizer to use to split the title and abstract
            into words or other kinds of tokens.
        token_indexers : Indexers used to define input token
            representations.
    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from: %s", file_path)
            json_data = json.load(data_file)
            for video_id, value in json_data.items():
                sentences = [postprocess(unidecode(x.strip()))
                             for x in value['sentences']]
                for first_sentence, second_sentence in pairwise(sentences):
                    yield self.text_to_instance(video_id, first_sentence, second_sentence)

    @overrides
    def text_to_instance(self,
                         video_id: str,
                         first_sentence: str,
                         second_sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_first_sentence = self._tokenizer.tokenize(first_sentence)
        tokenized_second_sentence = self._tokenizer.tokenize(second_sentence)
        first_sentence_field = TextField(tokenized_first_sentence, self._token_indexers)
        second_sentence_field = TextField(tokenized_second_sentence, self._token_indexers)
        fields = {'video_id': MetadataField(video_id),
                  'first_sentence': first_sentence_field,
                  'second_sentence': second_sentence_field}
        return Instance(fields)
