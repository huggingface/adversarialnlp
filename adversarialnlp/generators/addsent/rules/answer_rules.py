import math

from adversarialnlp.generators.addsent.utils import rejoin

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                    'august', 'september', 'october', 'november', 'december']

def ans_number(a, tokens, q, **kwargs):
    out_toks = []
    seen_num = False
    for t in tokens:
        ner = t['ner']
        pos = t['pos']
        w = t['word']
        out_tok = {'before': t['before']}

        # Split on dashes
        leftover = ''
        dash_toks = w.split('-')
        if len(dash_toks) > 1:
            w = dash_toks[0]
            leftover = '-'.join(dash_toks[1:])

        # Try to get a number out
        value = None
        if w != '%': 
            # Percent sign should just pass through
            try:
                value = float(w.replace(',', ''))
            except:
                try:
                    norm_ner = t['normalizedNER']
                    if norm_ner[0] in ('%', '>', '<'):
                        norm_ner = norm_ner[1:]
                    value = float(norm_ner)
                except:
                    pass
        if not value and (
                ner == 'NUMBER' or 
                (ner == 'PERCENT' and pos == 'CD')):
            # Force this to be a number anyways
            value = 10
        if value:
            if math.isinf(value) or math.isnan(value): value = 9001
            seen_num = True
            if w in ('thousand', 'million', 'billion', 'trillion'):
                if w == 'thousand':
                    new_val = 'million'
                else:
                    new_val = 'thousand'
            else:
                if value < 2500 and value > 1000:
                    new_val = str(value - 75)
                else:
                    # Change leading digit
                    if value == int(value):
                        val_chars = list('%d' % value)
                    else:
                        val_chars = list('%g' % value)
                    c = val_chars[0]
                    for i in range(len(val_chars)):
                        c = val_chars[i]
                        if c >= '0' and c <= '9':
                            val_chars[i] = str(max((int(c) + 5) % 10, 1))
                            break
                    new_val = ''.join(val_chars)
            if leftover:
                new_val = '%s-%s' % (new_val, leftover)
            out_tok['originalText'] = new_val
        else:
            out_tok['originalText'] = t['originalText']
        out_toks.append(out_tok)
    if seen_num:
        return rejoin(out_toks).strip()
    else:
        return None

def ans_date(a, tokens, q, **kwargs):
    out_toks = []
    if not all(t['ner'] == 'DATE' for t in tokens):
        return None
    for t in tokens:
        if t['pos'] == 'CD' or t['word'].isdigit():
            try:
                value = int(t['word'])
            except:
                value = 10  # fallback
            if value > 50:  new_val = str(value - 25)  # Year
            else:  # Day of month
                if value > 15: new_val = str(value - 11)
                else: new_val = str(value + 11)
        else:
            if t['word'].lower() in MONTHS:
                m_ind = MONTHS.index(t['word'].lower())
                new_val = MONTHS[(m_ind + 6) % 12].title()
            else:
                # Give up
                new_val = t['originalText']
        out_toks.append({'before': t['before'], 'originalText': new_val})
    new_ans = rejoin(out_toks).strip()
    if new_ans == a['text']: return None
    return new_ans

def ans_entity_full(ner_tag, new_ans):
    """Returns a function that yields new_ans iff every token has |ner_tag|."""
    def func(a, tokens, q, **kwargs):
        for t in tokens:
            if t['ner'] != ner_tag: return None
        return new_ans
    return func

def ans_abbrev(new_ans):
    def func(a, tokens, q, **kwargs):
        s = a['text']
        if s == s.upper() and s != s.lower():
            return new_ans
        return None
    return func

def ans_match_wh(wh_word, new_ans):
    """Returns a function that yields new_ans if the question starts with |wh_word|."""
    def func(a, tokens, q, **kwargs):
        if q.lower().startswith(wh_word + ' '):
            return new_ans
        return None
    return func

def ans_pos(pos, new_ans, end=False, add_dt=False):
    """Returns a function that yields new_ans if the first/last token has |pos|."""
    def func(a, tokens, q, determiner, **kwargs):
        if end:
            t = tokens[-1]
        else:
            t = tokens[0]
        if t['pos'] != pos: return None
        if add_dt and determiner:
            return '%s %s' % (determiner, new_ans)
        return new_ans
    return func

    
def ans_catch_all(new_ans):
    def func(a, tokens, q, **kwargs):
        return new_ans
    return func

ANSWER_RULES = [
        ('date', ans_date),
        ('number', ans_number),
        ('ner_person', ans_entity_full('PERSON', 'Jeff Dean')),
        ('ner_location', ans_entity_full('LOCATION', 'Chicago')),
        ('ner_organization', ans_entity_full('ORGANIZATION', 'Stark Industries')),
        ('ner_misc', ans_entity_full('MISC', 'Jupiter')),
        ('abbrev', ans_abbrev('LSTM')),
        ('wh_who', ans_match_wh('who', 'Jeff Dean')),
        ('wh_when', ans_match_wh('when', '1956')),
        ('wh_where', ans_match_wh('where', 'Chicago')),
        ('wh_where', ans_match_wh('how many', '42')),
        # Starts with verb
        ('pos_begin_vb', ans_pos('VB', 'learn')),
        ('pos_end_vbd', ans_pos('VBD', 'learned')),
        ('pos_end_vbg', ans_pos('VBG', 'learning')),
        ('pos_end_vbp', ans_pos('VBP', 'learns')),
        ('pos_end_vbz', ans_pos('VBZ', 'learns')),
        # Ends with some POS tag
        ('pos_end_nn', ans_pos('NN', 'hamster', end=True, add_dt=True)),
        ('pos_end_nnp', ans_pos('NNP', 'Central Park', end=True, add_dt=True)),
        ('pos_end_nns', ans_pos('NNS', 'hamsters', end=True, add_dt=True)),
        ('pos_end_nnps', ans_pos('NNPS', 'Kew Gardens', end=True, add_dt=True)),
        ('pos_end_jj', ans_pos('JJ', 'deep', end=True)),
        ('pos_end_jjr', ans_pos('JJR', 'deeper', end=True)),
        ('pos_end_jjs', ans_pos('JJS', 'deepest', end=True)),
        ('pos_end_rb', ans_pos('RB', 'silently', end=True)),
        ('pos_end_vbg', ans_pos('VBG', 'learning', end=True)),
        ('catch_all', ans_catch_all('aliens')),
]
