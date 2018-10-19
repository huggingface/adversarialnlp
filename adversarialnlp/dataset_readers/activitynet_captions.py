from typing import Dict
import json
import logging
import re
from itertools import tee
from overrides import overrides
from unidecode import unidecode
from num2words import num2words

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("activitynet_captions")
class ActivityNetCaptionsDatasetReader(DatasetReader):
    """
    Reads ActivityNet Captions JSON files and creates a dataset suitable for crafting
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

    The instances are created from all consecutive pair of sentences associated to each video.
    Ex: if a video has three associated sentences: s1, s2, s3
        read will generate two instances:
            1. Instance("first_sentence" = s1, "second_sentence" = s2)
            2. Instance("first_sentence" = s2, "second_sentence" = s3)

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the title and abstrct into words or other kinds of tokens.
        Defaults to ``WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
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
                sentences = [_postprocess(_remove_allcaps(unidecode(x.strip())))
                             for x in value['sentences']]
                for first_sentence, second_sentence in _pairwise(sentences):
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

def _pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def _n2w_1k(num, use_ordinal=False):
    if num > 1000:
        return ''
    return num2words(num, to='ordinal' if use_ordinal else 'cardinal')

def _postprocess(sentence):
    """
    make sure punctuation is followed by a space
    :param sentence:
    :return:
    """
    # Aggressively get rid of some punctuation markers
    sent0 = re.sub(r'^.*(\\|/|!!!|~|=|#|@|\*|¡|©|¿|«|»|¬|{|}|\||\(|\)|\+|\]|\[).*$',
                   ' ', sentence, flags=re.MULTILINE|re.IGNORECASE)

    # Less aggressively get rid of quotes, apostrophes
    sent1 = re.sub(r'"', ' ', sent0)
    sent2 = re.sub(r'`', '\'', sent1)

    # match ordinals
    sent3 = re.sub(r'(\d+(?:rd|st|nd))',
                   lambda x: _n2w_1k(int(x.group(0)[:-2]), use_ordinal=True), sent2)

    #These things all need to be followed by spaces or else we'll run into problems
    sent4 = re.sub(r'[:;,\"\!\.\-\?](?! )', lambda x: x.group(0) + ' ', sent3)

    #These things all need to be preceded by spaces or else we'll run into problems
    sent5 = re.sub(r'(?! )[-]', lambda x: ' ' + x.group(0), sent4)

    # Several spaces
    sent6 = re.sub(r'\s\s+', ' ', sent5)

    sent7 = sent6.strip()
    return sent7

def _remove_allcaps(sent):
    """
    Given a sentence, filter it so that it doesn't contain some words that are ALLcaps
    :param sent: string, like SOMEONE wheels SOMEONE on, mouthing silent words of earnest prayer.
    :return:                  Someone wheels someone on, mouthing silent words of earnest prayer.
    """
    # Remove all caps
    def _sanitize(word, is_first):
        if word == "I":
            return word
        num_capitals = len([x for x in word if not x.islower()])
        if num_capitals > len(word) // 2:
            # We have an all caps word here.
            if is_first:
                return word[0] + word[1:].lower()
            return word.lower()
        return word

    return ' '.join([_sanitize(word, i == 0) for i, word in enumerate(sent.split(' '))])
