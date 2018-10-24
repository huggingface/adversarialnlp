import logging
import json
import itertools
from typing import Iterable, Dict, Tuple
from collections import defaultdict

from adversarialnlp.common.file_utils import download_files
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

class AddSentGenerator():
    def __init__(self,
                 alteration_strategy: str = 'high-conf',
                 prepend: bool = False,
                 use_answer_placeholder: bool = False,
                 quiet: bool = False):
        """ Adversarial examples generator based on AddSent
            Possible strategies:
                - separate: Do best alteration for each word separately.
                - best: Generate exactly one best alteration (may over-alter).
                - high-conf: Do all possible high-confidence alterations
                - high-conf-separate: Do best high-confidence alteration for each word separately.
                - all: Do all possible alterations (very conservative)
        """
        super(AddSentGenerator).__init__()
        model_files = download_files(fnames=['nearby_n100_glove_6B_100d.json',
                                             'postag_dict.json'],
                                     model_folder='addsent')
        download_files(fnames=['train-v1.1.json', 'dev-v1.1.json'],
                       paths='https://rajpurkar.github.io/SQuAD-explorer/dataset/',
                       model_folder='squad')
        corenlp_path = download_files(fnames=['stanford-corenlp-full-2018-02-27.zip'],
                                      paths='http://nlp.stanford.edu/software/',
                                      model_folder='corenlp')

        self.nlp: StanfordCoreNLP = StanfordCoreNLP(corenlp_path[0])
        with open(model_files[0]) as data_file:
            self.nearby_word_dict: Dict = json.load(data_file)
        with open(model_files[1]) as data_file:
            self.postag_dict: Dict = json.load(data_file)

        self.alteration_strategy: str = alteration_strategy
        self.prepend: bool = prepend
        self.use_answer_placeholder: bool = use_answer_placeholder
        self.quiet: bool = quiet

        self._epochs: Dict[int, int] = defaultdict(int)

    def close(self):
        self.nlp.close()

    def _annotate(self, text: str, annotators: str):
        props = {'annotators': annotators,
                 'ssplit.newlineIsSentenceBreak': 'always',
                 'outputFormat':'json'}
        return json.loads(self.nlp.annotate(text, properties=props))

    def _alter_question(self, question, tokens, const_parse):
        """Alter the question to make it ask something else.
        """
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

    def edit_seed_instance(self, seed_instance: Tuple):
        """Edit a SQuAD example using rules.
        """
        qas, paragraph, title = seed_instance
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

    def __call__(self,
                 seed_instances: Iterable = None,
                 num_epochs: int = None,
                 shuffle: bool = True) -> Iterable:
        """
        Returns a generator that yields batches over the given dataset
        for the given number of epochs. If ``num_epochs`` is not specified,
        it will yield batches forever.
        Parameters
        ----------
        instances : ``Iterable[Instance]``
            The instances in the dataset. IMPORTANT: this must be able to be
            iterated over *multiple times*. That is, it must be either a List
            or some other object whose ``__iter__`` method returns a fresh iterator
            each time it's called.
        num_epochs : ``int``, optional (default=``None``)
            How times should we iterate over this dataset?  If ``None``, we will iterate over it
            forever.
        shuffle : ``bool``, optional (default=``True``)
            If ``True``, we will shuffle the instances in ``dataset`` before constructing batches
            and iterating over the data.
        """
        if seed_instances is None:
            seed_instances = squad_reader(SQUAD_FILE)
        # Instances is likely to be a list, which cannot be used as a key,
        # so we take the object id instead.
        key = id(seed_instances)
        starting_epoch = self._epochs[key]

        if num_epochs is None:
            epochs: Iterable[int] = itertools.count(starting_epoch)
        else:
            epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:
            self._epochs[key] = epoch
            for seed_instance in seed_instances:
                yield from self.edit_seed_instance(seed_instance)

# from adversarialnlp.common.file_utils import FIXTURES_ROOT
# generator = AddSentGenerator()
# test_instances = squad_reader(FIXTURES_ROOT / 'squad.json')
# batches = list(generator(test_instances, num_epochs=1))
