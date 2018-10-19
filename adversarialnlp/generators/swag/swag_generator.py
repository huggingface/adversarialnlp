# pylint: disable=invalid-name,arguments-differ
from typing import List, Iterable, Tuple
import logging

import numpy as np
from allennlp.predictors import Predictor
from allennlp.common.util import JsonDict
from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.models import Model
# from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.pretrained import span_based_constituency_parsing_with_elmo_joshi_2018, decomposable_attention_with_elmo_parikh_2017
from allennlp.models.archival import load_archive
from allennlp.models import BiMpm

from adversarialnlp.generators import Generator
from adversarialnlp.generators.swag.simple_bilm import SimpleBiLM

BATCH_SIZE = 1

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
        # self.language_model: Model = ElmoLstm()
        archive = load_archive('https://s3-us-west-2.amazonaws.com/allennlp/models/bimpm-quora-2018.08.17.tar.gz')
        # https://s3-us-west-2.amazonaws.com/allennlp/datasets/quora-question-paraphrase/test.tsv')
        vocab = archive.model.vocab
        self.language_model: Model = SimpleBiLM(vocab=vocab)
        # self.iterator = BasicIterator(batch_size=1)

    def __call__(self,
                 instances: Iterable[Instance]) -> Iterable[Instance]:
        """
        Returns a generator that yields batches of adversarial examples generated from
        the ``Instances``.
        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset to be used as seeds for generating the adversarial examples.
            IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        """
        generated_examples = []
        for instance in instances:
            first_sentence: TextField = instance.fields["first_sentence"]
            second_sentence: TextField = instance.fields["second_sentence"]
            eos_bounds = [i + 1 for i, x in enumerate(first_sentence.tokens) if x.text in ('.', '?', '!')]
            if not eos_bounds:
                first_sentence = TextField(tokens=first_sentence.tokens + [Token(text='.')],
                                           token_indexers=first_sentence.token_indexers)
            context_len = len(first_sentence.tokens)
            if context_len < 6 or context_len > 100:
                print("skipping on {} (too short or long)".format(
                        ' '.join(first_sentence.tokens + second_sentence.tokens)))
                continue
            # Something I should have done: make sure that there aren't multiple periods, etc. in s2 or in the middle
            eos_bounds_s2 = [i + 1 for i, x in enumerate(second_sentence.tokens) if x.text in ('.', '?', '!')]
            if len(eos_bounds_s2) > 1 or max(eos_bounds_s2) != len(second_sentence.tokens):
                continue
            elif not eos_bounds_s2:
                second_sentence = TextField(tokens=second_sentence.tokens + [Token(text='.')],
                                            token_indexers=second_sentence.token_indexers)

            # Now split on the VP
            startphrase, endphrase = self.split_on_final_vp(second_sentence)
            if startphrase is None or not startphrase or len(endphrase) < 5 or len(endphrase) > 25:
                print("skipping on {}->{},{}".format(' '.join(first_sentence.tokens + second_sentence.tokens),
                                                     startphrase, endphrase), flush=True)
                continue

            # if endphrase contains unk then it's hopeless
            # if any(vocab.get_token_index(tok.lower()) == vocab.get_token_index(vocab._oov_token)
            #        for tok in endphrase):
            #     print("skipping on {} (unk!)".format(' '.join(s1_toks + s2_toks)))
            #     continue

            context = [token.text for token in first_sentence.tokens] + startphrase

            gens0, fwd_scores, ctx_scores = self.language_model.conditional_generation(context, gt_completion=endphrase,
                                                                        batch_size=2 * BATCH_SIZE,
                                                                        max_gen_length=25)
            if len(gens0) < BATCH_SIZE:
                print("Couldnt generate enough candidates so skipping")
                continue
            gens0 = gens0[:BATCH_SIZE]
            fwd_scores = fwd_scores[:BATCH_SIZE]

            # Now get the backward scores.
            # full_sents = [context + gen for gen in gens0]  # NOTE: #1 is GT
            # result_dict = self.language_model(self.language_model.batch_to_ids(full_sents),
            #                                   use_forward=False, use_reverse=True, compute_logprobs=True)
            # ending_lengths = (fwd_scores < 0).sum(1)
            # ending_lengths_float = ending_lengths.astype(np.float32)
            # rev_scores = result_dict['reverse_logprobs'].cpu().detach().numpy()

            # forward_logperp_ending = -fwd_scores.sum(1) / ending_lengths_float
            # reverse_logperp_ending = -rev_scores[:, context_len:].sum(1) / ending_lengths_float
            # forward_logperp_begin = -ctx_scores.mean()
            # reverse_logperp_begin = -rev_scores[:, :context_len].mean(1)
            # eos_logperp = -fwd_scores[np.arange(fwd_scores.shape[0]), ending_lengths - 1]
            # # print("Time elapsed {:.3f}".format(time() - tic), flush=True)

            # scores = np.exp(np.column_stack((
            #     forward_logperp_ending,
            #     reverse_logperp_ending,
            #     reverse_logperp_begin,
            #     eos_logperp,
            #     np.ones(forward_logperp_ending.shape[0], dtype=np.float32) * forward_logperp_begin,
            # )))

            # # PRINTOUT
            # low2high = scores[:, 2].argsort()
            # print("\n\n Dataset={} ctx: {} (perp={:.3f})\n~~~\n".format(item['dataset'], ' '.join(context),
            #                                                             np.exp(forward_logperp_begin)), flush=True)
            # for i, ind in enumerate(low2high.tolist()):
            #     gen_i = ' '.join(gens0[ind])
            #     if (ind == 0) or (i < 128):
            #         print("{:3d}/{:4d}) ({}, end|ctx:{:5.1f} end:{:5.1f} ctx|end:{:5.1f} EOS|(ctx, end):{:5.1f}) {}".format(
            #             i, len(gens0), 'GOLD' if ind == 0 else '    ', *scores[ind][:-1], gen_i), flush=True)
            # gt_score = low2high.argsort()[0]

            # item_full = deepcopy(item)
            # item_full['sent1'] = s1_toks
            # item_full['startphrase'] = startphrase
            # item_full['context'] = context
            # item_full['generations'] = gens0
            # item_full['postags'] = [  # parse real fast
            #     [x.orth_.lower() if pos_vocab.get_token_index(x.orth_.lower()) != 1 else x.pos_ for x in y]
            #     for y in spacy_model.pipe([startphrase + gen for gen in gens0], batch_size=BATCH_SIZE)]
            # item_full['scores'] = pd.DataFrame(data=scores, index=np.arange(scores.shape[0]),
            #                                 columns=['end-from-ctx', 'end', 'ctx-from-end', 'eos-from-ctxend', 'ctx'])

            generated_examples.append(gens0)
            if len(generated_examples) > 0:
                yield generated_examples
                generated_examples = []

    def split_on_final_vp(self, sentence: Instance) -> (List[str], List[str]):
        """ Splits a sentence on the final verb phrase"""
        sentence_txt = ' '.join(t.text for t in sentence.tokens)
        res = self.constituency_predictor.predict(sentence_txt)
        res_chunked = self.find_VP(res['hierplane_tree']['root'])
        is_vp: List[int] = [i for i, (word, pos) in enumerate(res_chunked) if pos == 'VP']
        if not is_vp:
            return None, None
        vp_ind = max(is_vp)
        not_vp = [token for x in res_chunked[:vp_ind] for token in x[0].split(' ')]
        is_vp = [token for x in res_chunked[vp_ind:] for token in x[0].split(' ')]
        return not_vp, is_vp

    # We want to recurse until we find verb phrases
    def find_VP(self, tree: JsonDict) -> List[Tuple[str, any]]:
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
