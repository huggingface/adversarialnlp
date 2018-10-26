# pylint: disable=invalid-name,arguments-differ
from typing import List, Iterable, Tuple
import logging


import torch
from allennlp.common.util import JsonDict
from allennlp.common.file_utils import cached_path
from allennlp.data import Instance, Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.pretrained import PretrainedModel

from adversarialnlp.common.file_utils import download_files, DATA_ROOT
from adversarialnlp.generators import Generator
from adversarialnlp.generators.swag.simple_bilm import SimpleBiLM
from adversarialnlp.generators.swag.utils import optimistic_restore
from adversarialnlp.generators.swag.activitynet_captions_reader import ActivityNetCaptionsDatasetReader

BATCH_SIZE = 1
BEAM_SIZE = 8 * BATCH_SIZE

logger = logging.getLogger(__name__)

class SwagGenerator(Generator):
    """
    ``SwagGenerator`` inherit from the ``Generator`` class.
    This ``Generator`` generate adversarial examples from seeds using
    the method described in
    `SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference <http://arxiv.org/abs/1808.05326>`_.

    This method goes schematically as follows:
    - In a seed sample containing a pair of sequential sentence (ex: video captions),
        the second sentence is split into noun and verb phrases.
    - A language model generates several possible endings from the sencond sentence noun phrase.

    Args, input and yield:
        See the ``Generator`` class.

    Seeds:
        AllenNLP ``Instance`` containing two ``TextField``:
        `first_sentence` and `first_sentence`, respectively containing
        first and the second consecutive sentences.

    default_seeds:
        If no seeds are provided, the default_seeds are the training set
        of the
        `ActivityNet Captions dataset <https://cs.stanford.edu/people/ranjaykrishna/densevid/>`_.

    """
    def __init__(self,
                 default_seeds: Iterable = None,
                 quiet: bool = False):
        super().__init__(default_seeds, quiet)

        lm_files = download_files(fnames=['vocabulary.zip',
                                          'lm-fold-0.bin'],
                                  local_folder='swag_lm')

        activity_data_files = download_files(fnames=['captions.zip'],
                                             paths='https://cs.stanford.edu/people/ranjaykrishna/densevid/',
                                             local_folder='activitynet_captions')

        const_parser_files = cached_path('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz',
                             cache_dir=str(DATA_ROOT / 'allennlp_constituency_parser'))

        self.const_parser = PretrainedModel(const_parser_files, 'constituency-parser').predictor()
        vocab = Vocabulary.from_files(lm_files[0])
        self.language_model = SimpleBiLM(vocab=vocab, recurrent_dropout_probability=0.2,
                                         embedding_dropout_probability=0.2)
        optimistic_restore(self.language_model, torch.load(lm_files[1], map_location='cpu')['state_dict'])

        if default_seeds is None:
            self.default_seeds = ActivityNetCaptionsDatasetReader().read(activity_data_files[0] + '/train.json')
        else:
            self.default_seeds = default_seeds

    def _find_VP(self, tree: JsonDict) -> List[Tuple[str, any]]:
        r"""Recurse on a constituency parse tree until we find verb phrases"""

        # Recursion is annoying because we need to check whether each is a list or not
        def _recurse_on_children():
            assert 'children' in tree
            result = []
            for child in tree['children']:
                res = self._find_VP(child)
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

    def _split_on_final_vp(self, sentence: Instance) -> (List[str], List[str]):
        r"""Splits a sentence on the final verb phrase """
        sentence_txt = ' '.join(t.text for t in sentence.tokens)
        res = self.const_parser.predict(sentence_txt)
        res_chunked = self._find_VP(res['hierplane_tree']['root'])
        is_vp: List[int] = [i for i, (word, pos) in enumerate(res_chunked) if pos == 'VP']
        if not is_vp:
            return None, None
        vp_ind = max(is_vp)
        not_vp = [token for x in res_chunked[:vp_ind] for token in x[0].split(' ')]
        is_vp = [token for x in res_chunked[vp_ind:] for token in x[0].split(' ')]
        return not_vp, is_vp

    def generate_from_seed(self, seed: Tuple):
        """Edit a seed example.
        """
        first_sentence: TextField = seed.fields["first_sentence"]
        second_sentence: TextField = seed.fields["second_sentence"]
        eos_bounds = [i + 1 for i, x in enumerate(first_sentence.tokens) if x.text in ('.', '?', '!')]
        if not eos_bounds:
            first_sentence = TextField(tokens=first_sentence.tokens + [Token(text='.')],
                                        token_indexers=first_sentence.token_indexers)
        context_len = len(first_sentence.tokens)
        if context_len < 6 or context_len > 100:
            print("skipping on {} (too short or long)".format(
                    ' '.join(first_sentence.tokens + second_sentence.tokens)))
            return
        # Something I should have done: 
        # make sure that there aren't multiple periods, etc. in s2 or in the middle
        eos_bounds_s2 = [i + 1 for i, x in enumerate(second_sentence.tokens) if x.text in ('.', '?', '!')]
        if len(eos_bounds_s2) > 1 or max(eos_bounds_s2) != len(second_sentence.tokens):
            return
        elif not eos_bounds_s2:
            second_sentence = TextField(tokens=second_sentence.tokens + [Token(text='.')],
                                        token_indexers=second_sentence.token_indexers)

        # Now split on the VP
        startphrase, endphrase = self._split_on_final_vp(second_sentence)
        if startphrase is None or not startphrase or len(endphrase) < 5 or len(endphrase) > 25:
            print("skipping on {}->{},{}".format(' '.join(first_sentence.tokens + second_sentence.tokens),
                                                 startphrase, endphrase), flush=True)
            return

        # if endphrase contains unk then it's hopeless
        # if any(vocab.get_token_index(tok.lower()) == vocab.get_token_index(vocab._oov_token)
        #        for tok in endphrase):
        #     print("skipping on {} (unk!)".format(' '.join(s1_toks + s2_toks)))
        #     return

        context = [token.text for token in first_sentence.tokens] + startphrase

        lm_out = self.language_model.conditional_generation(context, gt_completion=endphrase,
                                                            batch_size=BEAM_SIZE,
                                                            max_gen_length=25)
        gens0, fwd_scores, ctx_scores = lm_out
        if len(gens0) < BATCH_SIZE:
            print("Couldn't generate enough candidates so skipping")
            return
        gens0 = gens0[:BATCH_SIZE]
        yield gens0
        # fwd_scores = fwd_scores[:BATCH_SIZE]

        # # Now get the backward scores.
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

        # PRINTOUT
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
        # item_full['sent1'] = first_sentence
        # item_full['startphrase'] = startphrase
        # item_full['context'] = context
        # item_full['generations'] = gens0
        # item_full['postags'] = [  # parse real fast
        #     [x.orth_.lower() if pos_vocab.get_token_index(x.orth_.lower()) != 1 else x.pos_ for x in y]
        #     for y in spacy_model.pipe([startphrase + gen for gen in gens0], batch_size=BATCH_SIZE)]
        # item_full['scores'] = pd.DataFrame(data=scores, index=np.arange(scores.shape[0]),
        #                                 columns=['end-from-ctx', 'end', 'ctx-from-end', 'eos-from-ctxend', 'ctx'])

        # generated_examples.append(gens0)
        # if len(generated_examples) > 0:
        #     yield generated_examples
        #     generated_examples = []

# from adversarialnlp.common.file_utils import FIXTURES_ROOT
# generator = SwagGenerator()
# test_instances = ActivityNetCaptionsDatasetReader().read(FIXTURES_ROOT / 'activitynet_captions.json')
# batches = list(generator(test_instances, num_epochs=1))
# assert len(batches) != 0
