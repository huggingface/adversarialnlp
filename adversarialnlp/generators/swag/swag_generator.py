# pylint: disable=invalid-name,arguments-differ
from typing import Dict, List, Iterator
import logging

from nltk import Tree
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance, DataIterator, Token
from allennlp.data.fields import TextField
from allennlp.pretrained import span_based_constituency_parsing_with_elmo_joshi_2018

from adversarialnlp.generators import Generator

logger = logging.getLogger(__name__)

@Generator.register("swag")
class SwagGenerator(Generator):
    """
    ``SwagGenerator`` create adversarial examples using the method described in
    `SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference <http://arxiv.org/abs/1808.05326>`_.
    The method goes schematically as follows:
    - For a pair of sequential sentence (ex: video captions), the second sentence is split into noun and verb phrases.
    - A language model generates many negative ending

    Parameters
    ----------
    num_examples : int, optional (default = 10)
        Number of adversarial examples to generate at each self.generate() call.
    """
    def __init__(self,
                 num_examples: int = 8) -> None:
        super().__init__(num_examples)
        # self.spacy_model = get_spacy_model("en_core_web_sm", pos_tags=True, parse=False, ner=False)
        # self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)
        self.constituency_predictor: Predictor = span_based_constituency_parsing_with_elmo_joshi_2018()

    def __call__(self,
                 instances: DataIterator) -> Iterator[Instance]:
        """
        Returns a generator that yields batches of adversarial examples generated from
        the ``Instances`` in the ``DataIterator``.
        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset to be used as seeds for generating the adversarial examples.
            IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        """
        for instance in instances:
            s1_toks = instance["first_sentence"].tokens
            s2_toks = instance["second_sentence"].tokens
            eos_bounds = [i + 1 for i, x in enumerate(s1_toks) if x.text in ('.', '?', '!')]
            if len(eos_bounds) == 0:
                s1_toks = TextField(tokens=s1_toks.tokens + [Token(text='.')], token_indexers=s1_toks.token_indexers)
            context_len = len(s1_toks)
            if context_len < 6 or context_len > 100:
                print("skipping on {} (too short or long)".format(' '.join(s1_toks + s2_toks)))
                continue
            # Something I should have done: make sure that there aren't multiple periods, etc. in s2 or in the middle
            eos_bounds_s2 = [i + 1 for i, x in enumerate(s2_toks) if x.text in ('.', '?', '!')]
            if len(eos_bounds_s2) > 1 or max(eos_bounds_s2) != len(s2_toks):
                continue
            elif len(eos_bounds_s2) == 0:
                s2_toks = TextField(tokens=s2_toks.tokens + [Token(text='.')], token_indexers=s2_toks.token_indexers)

            # Now split on the VP
            startphrase, endphrase = self.split_on_final_vp(s2_toks)
            if startphrase is None or not startphrase or len(endphrase) < 5 or len(endphrase) > 25:
                print("skipping on {}->{},{}".format(' '.join(s1_toks.tokens + s2_toks.tokens),
                      startphrase, endphrase), flush=True)
                continue

            # if endphrase contains unk then it's hopeless
            if any(vocab.get_token_index(tok.lower()) == vocab.get_token_index(vocab._oov_token) for tok in endphrase):
                print("skipping on {} (unk!)".format(' '.join(s1_toks + s2_toks)))
                continue

            context = s1_toks + startphrase

            gens0, fwd_scores, ctx_scores = model.conditional_generation(context, gt_completion=endphrase,
                                                                        batch_size=2 * BATCH_SIZE,
                                                                        max_gen_length=25)
            if len(gens0) < BATCH_SIZE:
                print("Couldnt generate enough candidates so skipping")
                continue
            yield contexts, endphrases

    def split_on_final_vp(self, sentence: Instance) -> (List[str], List[str]):
        """ Splits a sentence on the final verb phrase"""
        res = self.constituency_predictor.predict_instance(sentence)
        res_chunked = self.find_VP(res['hierplane_tree']['root'])
        is_vp: List[int] = [i for i, (word, pos) in enumerate(res_chunked) if pos == 'VP']
        if not is_vp:
            return None, None
        vp_ind = max(is_vp)
        not_vp = [token for x in res_chunked[:vp_ind] for token in x[0].split(' ')]
        is_vp = [token for x in res_chunked[vp_ind:] for token in x[0].split(' ')]
        return not_vp, is_vp

    # We want to recurse until we find verb phrases
    def find_VP(self, tree: JsonDict) -> List[(str, any)]:
        """
        Recurse on the tree until we find verb phrases
        :param tree: constituency parser result
        :return:
        """

        # Recursion is annoying because we need to check whether each is a list or not
        def _recurse_on_children():
            assert 'children' in tree
            result = []
            for child in tree['children']:
                res = self.find_VP(child)
                if isinstance(res, tuple):
                    result.append(res)
                else:
                    result.extend(res)
            return result

        if 'VP' in tree['attributes']:
            # # Now we'll get greedy and see if we can find something better
            # if 'children' in tree and len(tree['children']) > 1:
            #     recurse_result = _recurse_on_children()
            #     if all([x[1] in ('VP', 'NP', 'CC') for x in recurse_result]):
            #         return recurse_result
            return [(tree['word'], 'VP')]
        # base cases
        if 'NP' in tree['attributes']:
            return [(tree['word'], 'NP')]
        # No children
        if not 'children' in tree:
            return [(tree['word'], tree['attributes'][0])]

        # If a node only has 1 child then we'll have to stick with that
        if len(tree['children']) == 1:
            return _recurse_on_children()
        # try recursing on everything
        return _recurse_on_children()
