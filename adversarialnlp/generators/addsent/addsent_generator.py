import logging
import json
import itertools
from typing import Iterable, Dict, Tuple
from collections import defaultdict

from adversarialnlp.common.file_utils import download_files
from adversarialnlp.generators import Generator
from adversarialnlp.generators.addsent.rules import (ANSWER_RULES, HIGH_CONF_ALTER_RULES, ALL_ALTER_RULES,
                                                     DO_NOT_ALTER, BAD_ALTERATIONS, CONVERSION_RULES)
from adversarialnlp.generators.addsent.utils import (rejoin, ConstituencyParse, get_tokens_for_answers,
                                                     get_determiner_for_answers, read_const_parse)
from adversarialnlp.generators.addsent.squad_reader import squad_reader
from adversarialnlp.generators.addsent.corenlp import StanfordCoreNLP

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SQUAD_FILE = 'data/squad/train-v1.1.json'
NEARBY_GLOVE_FILE = 'data/addsent/nearby_n100_glove_6B_100d.json'
POSTAG_FILE = 'data/addsent/postag_dict.json'

class AddSentGenerator(Generator):
    r"""Adversarial examples generator based on AddSent.

    AddSent is described in the paper `Adversarial Examples for
    Evaluating Reading Comprehension Systems`_
    by Robin Jia & Percy Liang

    Args, input and yield:
        See the ``Generator`` class.

    Additional arguments:
        alteration_strategy: Alteration strategy. Options:

            - `separate`: Do best alteration for each word separately.
            - `best`: Generate exactly one best alteration
                (may over-alter).
            - `high-conf`: Do all possible high-confidence alterations.
            - `high-conf-separate`: Do best high-confidence alteration
                for each word separately.
            - `all`: Do all possible alterations (very conservative)

        prepend: Insert adversarial example at the beginning
            or end of the context.
        use_answer_placeholder: Use and answer placeholder.

    Seeds:
        Tuple of SQuAD-like instances containing
            - question-answer-span, and
            - context paragraph.

    default_seeds:
        If no seeds are provided, the default_seeds are the training
        set of the
        `SQuAD V1.0 dataset <https://rajpurkar.github.io/SQuAD-explorer/>`_.

    """
    def __init__(self,
                 alteration_strategy: str = 'high-conf',
                 prepend: bool = False,
                 use_answer_placeholder: bool = False,
                 default_seeds: Iterable = None,
                 quiet: bool = False):
        super(AddSentGenerator).__init__(default_seeds, quiet)
        model_files = download_files(fnames=['nearby_n100_glove_6B_100d.json',
                                             'postag_dict.json'],
                                     local_folder='addsent')
        corenlp_path = download_files(fnames=['stanford-corenlp-full-2018-02-27.zip'],
                                      paths='http://nlp.stanford.edu/software/',
                                      local_folder='corenlp')

        self.nlp: StanfordCoreNLP = StanfordCoreNLP(corenlp_path[0])
        with open(model_files[0], 'r') as data_file:
            self.nearby_word_dict: Dict = json.load(data_file)
        with open(model_files[1], 'r') as data_file:
            self.postag_dict: Dict = json.load(data_file)

        self.alteration_strategy: str = alteration_strategy
        self.prepend: bool = prepend
        self.use_answer_placeholder: bool = use_answer_placeholder
        if default_seeds is None:
            self.default_seeds = squad_reader(SQUAD_FILE)
        else:
            self.default_seeds = default_seeds

    def close(self):
        self.nlp.close()

    def _annotate(self, text: str, annotators: str):
        r"""Wrapper to call CoreNLP. """
        props = {'annotators': annotators,
                 'ssplit.newlineIsSentenceBreak': 'always',
                 'outputFormat':'json'}
        return json.loads(self.nlp.annotate(text, properties=props))

    def _alter_question(self, question, tokens, const_parse):
        r"""Alter the question to make it ask something else. """
        used_words = [tok['word'].lower() for tok in tokens]
        new_qs = []
        toks_all = []
        if self.alteration_strategy.startswith('high-conf'):
            rules = HIGH_CONF_ALTER_RULES
        else:
            rules = ALL_ALTER_RULES
        for i, tok in enumerate(tokens):
            if tok['word'].lower() in DO_NOT_ALTER:
                if self.alteration_strategy in ('high-conf', 'all'):
                    toks_all.append(tok)
                continue
            begin = tokens[:i]
            end = tokens[i+1:]
            found = False
            for rule_name in rules:
                rule = rules[rule_name]
                new_words = rule(tok, nearby_word_dict=self.nearby_word_dict, postag_dict=self.postag_dict)
                if new_words:
                    for word in new_words:
                        if word.lower() in used_words:
                            continue
                        if word.lower() in BAD_ALTERATIONS:
                            continue
                        # Match capitzliation
                        if tok['word'] == tok['word'].upper():
                            word = word.upper()
                        elif tok['word'] == tok['word'].title():
                            word = word.title()
                        new_tok = dict(tok)
                        new_tok['word'] = new_tok['lemma'] = new_tok['originalText'] = word
                        new_tok['altered'] = True
                        # NOTE: obviously this is approximate
                        if self.alteration_strategy.endswith('separate'):
                            new_tokens = begin + [new_tok] + end
                            new_q = rejoin(new_tokens)
                            tag = '%s-%d-%s' % (rule_name, i, word)
                            new_const_parse = ConstituencyParse.replace_words(
                                    const_parse, [tok['word'] for tok in new_tokens])
                            new_qs.append((new_q, new_tokens, new_const_parse, tag))
                            break
                        elif self.alteration_strategy in ('high-conf', 'all'):
                            toks_all.append(new_tok)
                            found = True
                            break
                if self.alteration_strategy in ('high-conf', 'all') and found:
                    break
            if self.alteration_strategy in ('high-conf', 'all') and not found:
                toks_all.append(tok)
        if self.alteration_strategy in ('high-conf', 'all'):
            new_q = rejoin(toks_all)
            new_const_parse = ConstituencyParse.replace_words(
                    const_parse, [tok['word'] for tok in toks_all])
            if new_q != question:
                new_qs.append((rejoin(toks_all), toks_all, new_const_parse, self.alteration_strategy))
        return new_qs

    def generate_from_seed(self, seed: Tuple):
        r"""Edit a SQuAD example using rules. """
        qas, paragraph = seed
        question = qas['question'].strip()
        if not self.quiet:
            print(f"Question: {question}")
        if self.use_answer_placeholder:
            answer = 'ANSWER'
            determiner = ''
        else:
            p_parse = self._annotate(paragraph, 'tokenize,ssplit,pos,ner,entitymentions')
            ind, a_toks = get_tokens_for_answers(qas['answers'], p_parse)
            determiner = get_determiner_for_answers(qas['answers'])
            answer_obj = qas['answers'][ind]
            for _, func in ANSWER_RULES:
                answer = func(answer_obj, a_toks, question, determiner=determiner)
                if answer:
                    break
            else:
                raise ValueError('Missing answer')
        q_parse = self._annotate(question, 'tokenize,ssplit,pos,parse,ner')
        q_parse = q_parse['sentences'][0]
        q_tokens = q_parse['tokens']
        q_const_parse = read_const_parse(q_parse['parse'])
        if self.alteration_strategy:
            # Easiest to alter the question before converting
            q_list = self._alter_question(question, q_tokens, q_const_parse)
        else:
            q_list = [(question, q_tokens, q_const_parse, 'unaltered')]
        for q_str, q_tokens, q_const_parse, tag in q_list:
            for rule in CONVERSION_RULES:
                sent = rule.convert(q_str, answer, q_tokens, q_const_parse)
                if sent:
                    if not self.quiet:
                        print(f"  Sent ({tag}): {sent}'")
                    cur_qa = {
                            'question': qas['question'],
                            'id': '%s-%s' % (qas['id'], tag),
                            'answers': qas['answers']
                    }
                    if self.prepend:
                        cur_text = '%s %s' % (sent, paragraph)
                        new_answers = []
                        for ans in qas['answers']:
                            new_answers.append({
                                    'text': ans['text'],
                                    'answer_start': ans['answer_start'] + len(sent) + 1
                            })
                        cur_qa['answers'] = new_answers
                    else:
                        cur_text = '%s %s' % (paragraph, sent)
                    out_example = {'title': title,
                                   'seed_context': paragraph,
                                   'seed_qas': qas,
                                   'context': cur_text,
                                   'qas': [cur_qa]}
                    yield out_example

# from adversarialnlp.common.file_utils import FIXTURES_ROOT
# generator = AddSentGenerator()
# test_instances = squad_reader(FIXTURES_ROOT / 'squad.json')
# batches = list(generator(test_instances, num_epochs=1))
# assert len(batches) != 0
