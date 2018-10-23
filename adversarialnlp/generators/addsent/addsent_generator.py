import json
from stanfordcorenlp import StanfordCoreNLP
from .answer_rules import ANSWER_RULES
from .alteration_rules import (HIGH_CONF_ALTER_RULES, ALL_ALTER_RULES,
                               DO_NOT_ALTER, BAD_ALTERATIONS)
from .conversion_rules import CONVERSION_RULES
from .utils import rejoin, ConstituencyParse, get_tokens_for_answers, get_determiner_for_answers, read_const_parse

NEARBY_GLOVE_FILE = 'data/addsent/nearby_n100_glove_6B_100d.json'
POSTAG_FILE = 'data/addsent/postag_dict.json'

class AddSentGenerator():
    """ Adversarial examples generator based on AddSent
    """
    def __init__(self, quiet=True, prepend=False):
        super(AddSentGenerator).__init__()
        self.quiet = quiet
        self.prepend = prepend
        self.nlp = StanfordCoreNLP(r'lib\stanford-corenlp-full-2018-02-27')
        with open(NEARBY_GLOVE_FILE) as data_file:
            self.nearby_word_dict = json.load(data_file)
        with open(POSTAG_FILE) as data_file:
            self.postag_dict = json.load(data_file)

    def alter_question(self, question, tokens, const_parse, strategy='separate'):
        """Alter the question to make it ask something else.

        Possible strategies:
            - separate: Do best alteration for each word separately.
            - best: Generate exactly one best alteration (may over-alter).
            - high-conf: Do all possible high-confidence alterations
            - high-conf-separate: Do best high-confidence alteration for each word separately.
            - all: Do all possible alterations (very conservative)
        """
        used_words = [tok['word'].lower() for tok in tokens]
        new_qs = []
        toks_all = []
        if strategy.startswith('high-conf'):
            rules = HIGH_CONF_ALTER_RULES
        else:
            rules = ALL_ALTER_RULES
        for i, tok in enumerate(tokens):
            if tok['word'].lower() in DO_NOT_ALTER:
                if strategy in ('high-conf', 'all'):
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
                        if strategy.endswith('separate'):
                            new_tokens = begin + [new_tok] + end
                            new_q = rejoin(new_tokens)
                            tag = '%s-%d-%s' % (rule_name, i, word)
                            new_const_parse = ConstituencyParse.replace_words(
                                    const_parse, [tok['word'] for tok in new_tokens])
                            new_qs.append((new_q, new_tokens, new_const_parse, tag))
                            break
                        elif strategy in ('high-conf', 'all'):
                            toks_all.append(new_tok)
                            found = True
                            break
                if strategy in ('high-conf', 'all') and found:
                    break
            if strategy in ('high-conf', 'all') and not found:
                toks_all.append(tok)
        if strategy in ('high-conf', 'all'):
            new_q = rejoin(toks_all)
            new_const_parse = ConstituencyParse.replace_words(
                    const_parse, [tok['word'] for tok in toks_all])
            if new_q != question:
                new_qs.append((rejoin(toks_all), toks_all, new_const_parse, strategy))
        return new_qs

    def _annotate(self, text, annotators):
        props = {'annotators': annotators,
                 'ssplit.newlineIsSentenceBreak': 'always',
                 'outputFormat':'json'}
        return json.loads(self.nlp.annotate(text, properties=props))

    def generate(self, articles, use_answer_placeholder=False, alteration_strategy=None):
        out_data = []
        for article in articles:
            out_paragraphs = []
            out_article = {'title': article['title'], 'paragraphs': out_paragraphs}
            out_data.append(out_article)
            for paragraph in article['paragraphs']:
                out_paragraphs.append(paragraph)
                for qa in paragraph['qas']:
                    question = qa['question'].strip()
                    if not self.quiet:
                        print(f"Question: {question}")
                    if use_answer_placeholder:
                        answer = 'ANSWER'
                        determiner = ''
                    else:
                        p_parse = self._annotate(paragraph['context'], 'tokenize,ssplit,pos,ner,entitymentions')
                        ind, a_toks = get_tokens_for_answers(qa['answers'], p_parse)
                        determiner = get_determiner_for_answers(qa['answers'])
                        answer_obj = qa['answers'][ind]
                        for rule_name, func in ANSWER_RULES:
                            answer = func(answer_obj, a_toks, question, determiner=determiner)
                            if answer:
                                break
                        else:
                            raise ValueError('Missing answer')
                    q_parse = self._annotate(question, 'tokenize,ssplit,pos,parse,ner')
                    q_tokens = q_parse['sentences'][0]['tokens']
                    q_const_parse = read_const_parse(q_parse['parse'])
                    if alteration_strategy:
                        # Easiest to alter the question before converting
                        q_list = self.alter_question(question, q_tokens, q_const_parse,
                                                     strategy=alteration_strategy)
                    else:
                        q_list = [(question, q_tokens, q_const_parse, 'unaltered')]
                    for q_str, q_tokens, q_const_parse, tag in q_list:
                        for rule in CONVERSION_RULES:
                            sent = rule.convert(q_str, answer, q_tokens, q_const_parse)
                            if sent:
                                if not self.quiet:
                                    print("  Sent ({tag}): {sent}'")
                                cur_qa = {
                                        'question': qa['question'],
                                        'id': '%s-%s' % (qa['id'], tag),
                                        'answers': qa['answers']
                                }
                                if self.prepend:
                                    cur_text = '%s %s' % (sent, paragraph['context'])
                                    new_answers = []
                                    for a in qa['answers']:
                                        new_answers.append({
                                                'text': a['text'],
                                                'answer_start': a['answer_start'] + len(sent) + 1
                                        })
                                    cur_qa['answers'] = new_answers
                                else:
                                    cur_text = '%s %s' % (paragraph['context'], sent)
                                cur_paragraph = {'context': cur_text, 'qas': [cur_qa]}
                                out_paragraphs.append(cur_paragraph)
                                break
        return out_data
