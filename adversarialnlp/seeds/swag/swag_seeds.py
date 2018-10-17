# pylint: disable=invalid-name,arguments-differ
from typing import Dict, Tuple, List, Optional, NamedTuple, Any
import logging
from overrides import overrides

import torch
from tqdm import tqdm
from spacy.tokens import Doc
from allennlp.commands.predict import Predictor
from allennlp.common.util import get_spacy_model
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.pretrained import span_based_constituency_parsing_with_elmo_joshi_2018
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from adversarialnlp.seeds.seeds_generator import SeedsGenerator

logger = logging.getLogger(__name__)

class SwagSeeds(SeedsGenerator):
    """
    ``SwagSeeds`` prepare ``Seeds`` to be used by ``SwagEditor`` to create an adversarial example.
    ``Seeds`` are created from examples sampled from the dataset.

    Parameters
    ----------
    dataset_reader : ``DatasetReader``
        The ``DatasetReader`` object that will be used to sample examples for preparing ``Seeds``
        objects.
    num_seeds : int, optional (default = 10)
        Number of seeds to generate at each self.generate() call.
    """
    def __init__(self,
                 dataset_reader: DatasetReader,
                 num_seeds: int = 10) -> None:
        super().__init__(dataset_reader)
        # self.spacy_model = get_spacy_model("en_core_web_sm", pos_tags=True, parse=False, ner=False)
        self._tokenizer = SpacyWordSplitter(language='en_core_web_sm', pos_tags=True)
        self.constituency_predictor = span_based_constituency_parsing_with_elmo_joshi_2018()
        self.num_seeds = num_seeds

        # TODO: What's this !!!
        # self.spacy_model.tokenizer = lambda x: Doc(self.spacy_model.vocab, x)
        # This is hella hacky! but it's tokenized already
        # self.constituency_predictor._tokenizer.spacy.tokenizer = lambda x: Doc(self.constituency_predictor._tokenizer.spacy.vocab, x)

    # We want to recurse until we find verb phrases
    def find_VP(self, tree):
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

    def split_on_final_vp(self, sentence):
        """ Splits a sentence on the final verb phrase"""
        try:
            res = self.constituency_predictor.predict_json({'sentence': sentence})
        except:
            return None, None
        res_chunked = self.find_VP(res['hierplane_tree']['root'])
        is_vp = [i for i, (word, pos) in enumerate(res_chunked) if pos == 'VP']
        if len(is_vp) == 0:
            return None, None
        vp_ind = max(is_vp)
        not_vp = [token for x in res_chunked[:vp_ind] for token in x[0].split(' ')]
        is_vp = [token for x in res_chunked[vp_ind:] for token in x[0].split(' ')]
        return not_vp, is_vp

    def generate(self,
                 num_seeds: int = -1) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        num_seeds : int, optional (default = -1)
            Number of seeds to generate.
            If -1, takes the number defined during ``SwagSeeds`` class initialization
        Returns
        -------
        An output dictionary consisting of:
        loss : ``torch.FloatTensor``, optional
            A scalar loss to be optimised.
        arc_loss : ``torch.FloatTensor``
            The loss contribution from the unlabeled arcs.
        loss : ``torch.FloatTensor``, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : ``torch.FloatTensor``
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : ``torch.FloatTensor``
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : ``torch.LongTensor``
            A mask denoting the padded elements in the batch.
        """
        if num_seeds == -1:
            num_seeds = self.num_seeds

        embedded_text_input = self.text_field_embedder(words)
        for (instance, s1_toks, s2_toks, item) in tqdm(stories_tokenized):

            eos_bounds = [i + 1 for i, x in enumerate(s1_toks) if x in ('.', '?', '!')]
            if len(eos_bounds) == 0:
                s1_toks.append('.')  # Just in case there's no EOS indicator.

            context_len = len(s1_toks)
            if context_len < 6 or context_len > 100:
                print("skipping on {} (too short or long)".format(' '.join(s1_toks + s2_toks)))
                continue

            # Something I should have done: make sure that there aren't multiple periods, etc. in s2 or in the middle
            eos_bounds_s2 = [i + 1 for i, x in enumerate(s2_toks) if x in ('.', '?', '!')]
            if len(eos_bounds_s2) > 1 or max(eos_bounds_s2) != len(s2_toks):
                continue
            elif len(eos_bounds_s2) == 0:
                s2_toks.append('.')


            # Now split on the VP
            startphrase, endphrase = self.split_on_final_vp(s2_toks)
            if startphrase is None or len(startphrase) == 0 or len(endphrase) < 5 or len(endphrase) > 25:
                print("skipping on {}->{},{}".format(' '.join(s1_toks + s2_toks), startphrase, endphrase), flush=True)
                continue

            # if endphrase contains unk then it's hopeless
            if any(vocab.get_token_index(tok.lower()) == vocab.get_token_index(vocab._oov_token) for tok in endphrase):
                print("skipping on {} (unk!)".format(' '.join(s1_toks + s2_toks)))
                continue

            context = s1_toks + startphrase
        
        return contexts, endphrases
