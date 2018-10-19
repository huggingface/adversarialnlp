# pylint: disable=no-self-use,invalid-name
from typing import List
import pytest

from allennlp.data.fields import TextField
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import _LazyInstances

from adversarialnlp.generators.swag.swag_generator import SwagGenerator
from adversarialnlp.dataset_readers import ActivityNetCaptionsDatasetReader
from adversarialnlp.tests.utils import FIXTURES_ROOT


class GeneratorTest(AllenNlpTestCase):
    def setUp(self):
        super(GeneratorTest, self).setUp()
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}
        self.vocab = Vocabulary()
        self.this_index = self.vocab.add_token_to_namespace('this')
        self.is_index = self.vocab.add_token_to_namespace('is')
        self.a_index = self.vocab.add_token_to_namespace('a')
        self.sentence_index = self.vocab.add_token_to_namespace('sentence')
        self.another_index = self.vocab.add_token_to_namespace('another')
        self.yet_index = self.vocab.add_token_to_namespace('yet')
        self.very_index = self.vocab.add_token_to_namespace('very')
        self.long_index = self.vocab.add_token_to_namespace('long')
        instances = [
                self.create_instance(["this", "is", "a", "sentence"], ["this", "is", "another", "sentence"]),
                self.create_instance(["yet", "another", "sentence"],
                                     ["this", "is", "a", "very", "very", "very", "very", "long", "sentence"]),
                ]

        class LazyIterable:
            def __iter__(self):
                return (instance for instance in instances)

        self.instances = instances
        self.lazy_instances = LazyIterable()

    def create_instance(self, first_sentence: List[str], second_sentence: List[str]):
        first_tokens = [Token(t) for t in first_sentence]
        second_tokens = [Token(t) for t in second_sentence]
        instance = Instance({'first_sentence': TextField(first_tokens, self.token_indexers),
                             'second_sentence': TextField(second_tokens, self.token_indexers)})
        return instance

    def assert_instances_are_correct(self, candidate_instances):
        # First we need to remove padding tokens from the candidates.
        # pylint: disable=protected-access
        candidate_instances = [tuple(w for w in instance if w != 0) for instance in candidate_instances]
        expected_instances = [tuple(instance.fields["first_sentence"]._indexed_tokens["tokens"])
                              for instance in self.instances]
        assert set(candidate_instances) == set(expected_instances)


class TestSwagGenerator(GeneratorTest):
    # The Generator should work the same for lazy and non lazy datasets,
    # so each remaining test runs over both.
    def test_yield_one_epoch_generation_over_the_data_once(self):
        for test_instances in (self.instances, self.lazy_instances):
            generator = SwagGenerator(num_examples=1)
            test_instances = ActivityNetCaptionsDatasetReader().read(FIXTURES_ROOT / 'activitynet_captions.json')
            batches = list(generator(test_instances))
            # We just want to get the single-token array for the text field in the instance.
            instances = [tuple(instance.detach().cpu().numpy())
                         for batch in batches
                         for instance in batch['text']["tokens"]]
            assert len(instances) == 5
            self.assert_instances_are_correct(instances)
