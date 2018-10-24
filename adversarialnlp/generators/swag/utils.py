import re
from itertools import tee

from num2words import num2words

def optimistic_restore(network, state_dict):
    mismatch = False
    own_state = network.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            print("Unexpected key {} in state_dict with size {}".format(name, param.size()))
            mismatch = True
        elif param.size() == own_state[name].size():
            own_state[name].copy_(param)
        else:
            print("Network has {} with size {}, ckpt has {}".format(name,
                                                                    own_state[name].size(),
                                                                    param.size()))
            mismatch = True

    missing = set(own_state.keys()) - set(state_dict.keys())
    if len(missing) > 0:
        print("We couldn't find {}".format(','.join(missing)))
        mismatch = True
    return not mismatch

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def n2w_1k(num, use_ordinal=False):
    if num > 1000:
        return ''
    return num2words(num, to='ordinal' if use_ordinal else 'cardinal')

def postprocess(sentence):
    """
    make sure punctuation is followed by a space
    :param sentence:
    :return:
    """
    sentence = remove_allcaps(sentence)
    # Aggressively get rid of some punctuation markers
    sent0 = re.sub(r'^.*(\\|/|!!!|~|=|#|@|\*|¡|©|¿|«|»|¬|{|}|\||\(|\)|\+|\]|\[).*$',
                   ' ', sentence, flags=re.MULTILINE|re.IGNORECASE)

    # Less aggressively get rid of quotes, apostrophes
    sent1 = re.sub(r'"', ' ', sent0)
    sent2 = re.sub(r'`', '\'', sent1)

    # match ordinals
    sent3 = re.sub(r'(\d+(?:rd|st|nd))',
                   lambda x: n2w_1k(int(x.group(0)[:-2]), use_ordinal=True), sent2)

    #These things all need to be followed by spaces or else we'll run into problems
    sent4 = re.sub(r'[:;,\"\!\.\-\?](?! )', lambda x: x.group(0) + ' ', sent3)

    #These things all need to be preceded by spaces or else we'll run into problems
    sent5 = re.sub(r'(?! )[-]', lambda x: ' ' + x.group(0), sent4)

    # Several spaces
    sent6 = re.sub(r'\s\s+', ' ', sent5)

    sent7 = sent6.strip()
    return sent7

def remove_allcaps(sent):
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
