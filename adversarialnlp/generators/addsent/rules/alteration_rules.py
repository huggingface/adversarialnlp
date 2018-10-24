import collections
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer

STEMMER = LancasterStemmer()

POS_TO_WORDNET = {
        'NN': wn.NOUN,
        'JJ': wn.ADJ,
        'JJR': wn.ADJ,
        'JJS': wn.ADJ,
}

def alter_special(token, **kwargs):
    w = token['originalText']
    if w in SPECIAL_ALTERATIONS:
        return [SPECIAL_ALTERATIONS[w]]
    return None

def alter_nearby(pos_list, ignore_pos=False, is_ner=False):
    def func(token, nearby_word_dict=None, postag_dict=None, **kwargs):
        if token['pos'] not in pos_list: return None
        if is_ner and token['ner'] not in ('PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'):
            return None
        w = token['word'].lower()
        if w in ('war'): return None
        if w not in nearby_word_dict: return None
        new_words = []
        w_stem = STEMMER.stem(w.replace('.', ''))
        for x in nearby_word_dict[w][1:]:
            new_word = x['word']
            # Make sure words aren't too similar (e.g. same stem)
            new_stem = STEMMER.stem(new_word.replace('.', ''))
            if w_stem.startswith(new_stem) or new_stem.startswith(w_stem): continue
            if not ignore_pos:
                # Check for POS tag match
                if new_word not in postag_dict: continue
                new_postag = postag_dict[new_word]
                if new_postag != token['pos']: continue 
            new_words.append(new_word)
        return new_words
    return func

def alter_entity_glove(token, nearby_word_dict=None, **kwargs):
    # NOTE: Deprecated
    if token['ner'] not in ('PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'): return None
    w = token['word'].lower()
    if w == token['word']: return None  # Only do capitalized words
    if w not in nearby_word_dict: return None
    new_words = []
    for x in nearby_word_dict[w][1:3]:
        if token['word'] == w.upper():
            new_words.append(x['word'].upper())
        else:
            new_words.append(x['word'].title())
    return new_words

def alter_entity_type(token, **kwargs):
    pos = token['pos']
    ner = token['ner']
    word = token['word']
    is_abbrev = word == word.upper() and not word == word.lower()
    if token['pos'] not in (
            'JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):
        # Don't alter non-content words
        return None
    if ner == 'PERSON':
        return ['Jackson']
    elif ner == 'LOCATION':
        return ['Berlin']
    elif ner == 'ORGANIZATION':
        if is_abbrev: return ['UNICEF']
        return ['Acme']
    elif ner == 'MISC':
        return ['Neptune']
    elif ner == 'NNP':
        if is_abbrev: return ['XKCD']
        return ['Dalek']
    elif pos == 'NNPS':
        return ['Daleks']
    return None

def alter_wordnet_antonyms(token, **kwargs):
    if token['pos'] not in POS_TO_WORDNET: return None
    w = token['word'].lower()
    wn_pos = POS_TO_WORDNET[token['pos']]
    synsets = wn.synsets(w, wn_pos)
    if not synsets: return None
    synset = synsets[0]
    antonyms = []
    for lem in synset.lemmas():
        if lem.antonyms():
            for a in lem.antonyms():
                new_word = a.name()
                if '_' in a.name(): continue
                antonyms.append(new_word)
    return antonyms

SPECIAL_ALTERATIONS = {
        'States': 'Kingdom',
        'US': 'UK',
        'U.S': 'U.K.',
        'U.S.': 'U.K.',
        'UK': 'US',
        'U.K.': 'U.S.',
        'U.K': 'U.S.',
        'largest': 'smallest',
        'smallest': 'largest',
        'highest': 'lowest',
        'lowest': 'highest',
        'May': 'April',
        'Peyton': 'Trevor',
}

DO_NOT_ALTER = ['many', 'such', 'few', 'much', 'other', 'same', 'general',
                                'type', 'record', 'kind', 'sort', 'part', 'form', 'terms', 'use',
                                'place', 'way', 'old', 'young', 'bowl', 'united', 'one',
                                'likely', 'different', 'square', 'war', 'republic', 'doctor', 'color']

BAD_ALTERATIONS = ['mx2004', 'planet', 'u.s.', 'Http://Www.Co.Mo.Md.Us']

HIGH_CONF_ALTER_RULES = collections.OrderedDict([
        ('special', alter_special),
        ('wn_antonyms', alter_wordnet_antonyms),
        ('nearbyNum', alter_nearby(['CD'], ignore_pos=True)),
        ('nearbyProperNoun', alter_nearby(['NNP', 'NNPS'])),
        ('nearbyProperNoun', alter_nearby(['NNP', 'NNPS'], ignore_pos=True)),
        ('nearbyEntityNouns', alter_nearby(['NN', 'NNS'], is_ner=True)),
        ('nearbyEntityJJ', alter_nearby(['JJ', 'JJR', 'JJS'], is_ner=True)),
        ('entityType', alter_entity_type),
        #('entity_glove', alter_entity_glove),
])
ALL_ALTER_RULES = collections.OrderedDict(list(HIGH_CONF_ALTER_RULES.items()) + [
        ('nearbyAdj', alter_nearby(['JJ', 'JJR', 'JJS'])),
        ('nearbyNoun', alter_nearby(['NN', 'NNS'])),
        #('nearbyNoun', alter_nearby(['NN', 'NNS'], ignore_pos=True)),
])
