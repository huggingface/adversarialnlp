from pattern import en as patten

CONST_PARSE_MACROS = {
        '$Noun': '$NP/$NN/$NNS/$NNP/$NNPS',
        '$Verb': '$VB/$VBD/$VBP/$VBZ',
        '$Part': '$VBN/$VG',
        '$Be': 'is/are/was/were',
        '$Do': "do/did/does/don't/didn't/doesn't",
        '$WHP': '$WHADJP/$WHADVP/$WHNP/$WHPP',
}

# Map to pattern.en aliases
# http://www.clips.ua.ac.be/pages/pattern-en#conjugation
POS_TO_PATTERN = {
        'vb': 'inf',  # Infinitive
        'vbp': '1sg',  # non-3rd-person singular present
        'vbz': '3sg',  # 3rd-person singular present
        'vbg': 'part',  # gerund or present participle
        'vbd': 'p',  # past
        'vbn': 'ppart',  # past participle
}
# Tenses prioritized by likelihood of arising
PATTERN_TENSES = ['inf', '3sg', 'p', 'part', 'ppart', '1sg']

def _check_match(node, pattern_tok):
    if pattern_tok in CONST_PARSE_MACROS:
        pattern_tok = CONST_PARSE_MACROS[pattern_tok]
    if ':' in pattern_tok:
        # ':' means you match the LHS category and start with something on the right
        lhs, rhs = pattern_tok.split(':')
        match_lhs = _check_match(node, lhs)
        if not match_lhs: return False
        phrase = node.get_phrase().lower()
        retval = any(phrase.startswith(w) for w in rhs.split('/'))
        return retval
    elif '/' in pattern_tok:
        return any(_check_match(node, t) for t in pattern_tok.split('/'))
    return ((pattern_tok.startswith('$') and pattern_tok[1:] == node.tag) or
                    (node.word and pattern_tok.lower() == node.word.lower()))

def _recursive_match_pattern(pattern_toks, stack, matches):
    """Recursively try to match a pattern, greedily."""
    if len(matches) == len(pattern_toks):
        # We matched everything in the pattern; also need stack to be empty
        return len(stack) == 0
    if len(stack) == 0: return False
    cur_tok = pattern_toks[len(matches)]
    node = stack.pop()
    # See if we match the current token at this level
    is_match = _check_match(node, cur_tok)
    if is_match:
        cur_num_matches = len(matches)
        matches.append(node)
        new_stack = list(stack)
        success = _recursive_match_pattern(pattern_toks, new_stack, matches)
        if success: return True
        # Backtrack
        while len(matches) > cur_num_matches:
            matches.pop()
    # Recurse to children
    if not node.children: return False  # No children to recurse on, we failed
    stack.extend(node.children[::-1])  # Leftmost children should be popped first
    return _recursive_match_pattern(pattern_toks, stack, matches)

def match_pattern(pattern, const_parse):
    pattern_toks = pattern.split(' ')
    whole_phrase = const_parse.get_phrase()
    if whole_phrase.endswith('?') or whole_phrase.endswith('.'):
        # Match trailing punctuation as needed
        pattern_toks.append(whole_phrase[-1])
    matches = []
    success = _recursive_match_pattern(pattern_toks, [const_parse], matches)
    if success:
        return matches
    else:
        return None

def run_postprocessing(s, rules, all_args):
    rule_list = rules.split(',')
    for rule in rule_list:
        if rule == 'lower':
            s = s.lower()
        elif rule.startswith('tense-'):
            ind = int(rule[6:])
            orig_vb = all_args[ind]
            tenses = patten.tenses(orig_vb)
            for tense in PATTERN_TENSES:  # Prioritize by PATTERN_TENSES
                if tense in tenses:
                    break
            else:  # Default to first tense
                tense = PATTERN_TENSES[0]
            s = patten.conjugate(s, tense)
        elif rule in POS_TO_PATTERN:
            s = patten.conjugate(s, POS_TO_PATTERN[rule])
    return s

def convert_whp(node, q, a, tokens, quiet=False):
    if node.tag in ('WHNP', 'WHADJP', 'WHADVP', 'WHPP'):
        # Apply WHP rules
        cur_phrase = node.get_phrase()
        cur_tokens = tokens[node.get_start_index():node.get_end_index()]
        for r in WHP_RULES:
            phrase = r.convert(cur_phrase, a, cur_tokens, node, run_fix_style=False)
            if phrase:
                if not quiet:
                    print(f"  WHP Rule '{r.name}': {phrase}")
                return phrase
    return None

### Rules for converting questions into declarative sentences
def fix_style(s):
    """Minor, general style fixes for questions."""
    s = s.replace('?', '')  # Delete question marks anywhere in sentence.
    s = s.strip(' .')
    if s[0] == s[0].lower():
        s = s[0].upper() + s[1:]
    return s + '.'

class ConversionRule(object):
    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        raise NotImplementedError

class ConstituencyRule(ConversionRule):
    """A rule for converting question to sentence based on constituency parse."""
    def __init__(self, in_pattern, out_pattern, postproc=None):
        self.in_pattern = in_pattern   # e.g. "where did $NP $VP"
        self.out_pattern = out_pattern #unicode(out_pattern)
                # e.g. "{1} did {2} at {0}."  Answer is always 0
        self.name = in_pattern
        if postproc:
            self.postproc = postproc
        else:
            self.postproc = {}

    def convert(self, q, a, tokens, const_parse, run_fix_style=True) -> str:
        pattern_toks = self.in_pattern.split(' ')   # Don't care about trailing punctuation
        match = match_pattern(self.in_pattern, const_parse)
        appended_clause = False
        if not match:
            # Try adding a PP at the beginning
            appended_clause = True
            new_pattern = '$PP , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = match_pattern(new_pattern, const_parse)
        if not match:
            # Try adding an SBAR at the beginning
            new_pattern = '$SBAR , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = match_pattern(new_pattern, const_parse)
        if not match: return None
        appended_clause_match = None
        fmt_args = [a]
        for t, m in zip(pattern_toks, match):
            if t.startswith('$') or '/' in t:
                # First check if it's a WHP
                phrase = convert_whp(m, q, a, tokens)
                if not phrase:
                    phrase = m.get_phrase()
                fmt_args.append(phrase)
        if appended_clause:
            appended_clause_match = fmt_args[1]
            fmt_args = [a] + fmt_args[2:]
        for i in range(len(fmt_args)):
            if i in self.postproc:
                # Run postprocessing filters
                fmt_args[i] = run_postprocessing(fmt_args[i], self.postproc[i], fmt_args)
        output = self.gen_output(fmt_args)
        if appended_clause:
            output = appended_clause_match + ', ' + output
        if run_fix_style:
            output = fix_style(output)
        return output


    def gen_output(self, fmt_args):
        """By default, use self.out_pattern.  Can be overridden."""
        return self.out_pattern.format(*fmt_args)

class ReplaceRule(ConversionRule):
    """A simple rule that replaces some tokens with the answer."""
    def __init__(self, target, replacement='{}', start=False):
        self.target = target
        self.replacement = replacement #unicode(replacement)
        self.name = 'replace(%s)' % target
        self.start = start

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        t_toks = self.target.split(' ')
        q_toks = q.rstrip('?.').split(' ')
        replacement_text = self.replacement.format(a)
        for i in range(len(q_toks)):
            if self.start and i != 0: continue
            if ' '.join(q_toks[i:i + len(t_toks)]).rstrip(',').lower() == self.target:
                begin = q_toks[:i]
                end = q_toks[i + len(t_toks):]
                output = ' '.join(begin + [replacement_text] + end)
                if run_fix_style:
                    output = fix_style(output)
                return output
        return None

class FindWHPRule(ConversionRule):
    """A rule that looks for $WHP's from right to left and does replacements."""
    name = 'FindWHP'
    def _recursive_convert(self, node, q, a, tokens, found_whp):
        if node.word:
            return node.word, found_whp
        if not found_whp:
            whp_phrase = convert_whp(node, q, a, tokens)
            if whp_phrase:
                return whp_phrase, True
        child_phrases = []
        for c in node.children[::-1]:
            c_phrase, found_whp = self._recursive_convert(c, q, a, tokens, found_whp)
            child_phrases.append(c_phrase)
        out_toks = []
        for i, p in enumerate(child_phrases[::-1]):
            if i == 0 or p.startswith("'"):
                out_toks.append(p)
            else:
                out_toks.append(' ' + p)
        return ''.join(out_toks), found_whp

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        out_phrase, found_whp = self._recursive_convert(const_parse, q, a, tokens, False)
        if found_whp:
            if run_fix_style:
                out_phrase = fix_style(out_phrase)
            return out_phrase
        return None

class AnswerRule(ConversionRule):
    """Just return the answer."""
    name = 'AnswerRule'
    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        return a

CONVERSION_RULES = [
        # Special rules
        ConstituencyRule('$WHP:what $Be $NP called that $VP', '{2} that {3} {1} called {1}'),

        # What type of X
        #ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP/$Noun $Be $NP", '{5} {4} a {1} {3}'),
        #ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP/$Noun $Be $VP", '{1} {3} {4} {5}'),
        #ConstituencyRule("$WHP:what/which type/sort/kind/group of $NP $VP", '{1} {3} {4}'),

        # How $JJ
        ConstituencyRule('how $JJ $Be $NP $IN $NP', '{3} {2} {0} {1} {4} {5}'),
        ConstituencyRule('how $JJ $Be $NP $SBAR', '{3} {2} {0} {1} {4}'),
        ConstituencyRule('how $JJ $Be $NP', '{3} {2} {0} {1}'),

        # When/where $Verb
        ConstituencyRule('$WHP:when/where $Do $NP', '{3} occurred in {1}'),
        ConstituencyRule('$WHP:when/where $Do $NP $Verb', '{3} {4} in {1}', {4: 'tense-2'}),
        ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP/$PP', '{3} {4} {5} in {1}', {4: 'tense-2'}),
        ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP $PP', '{3} {4} {5} {6} in {1}', {4: 'tense-2'}),
        ConstituencyRule('$WHP:when/where $Be $NP', '{3} {2} in {1}'),
        ConstituencyRule('$WHP:when/where $Verb $NP $VP/$ADJP', '{3} {2} {4} in {1}'),

        # What/who/how $Do
        ConstituencyRule("$WHP:what/which/who $Do $NP do", '{3} {1}', {0: 'tense-2'}),
        ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb", '{3} {4} {1}', {4: 'tense-2'}),
        ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $IN/$NP", '{3} {4} {5} {1}', {4: 'tense-2', 0: 'vbg'}),
        ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $PP", '{3} {4} {1} {5}', {4: 'tense-2', 0: 'vbg'}),
        ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $NP $VP", '{3} {4} {5} {6} {1}', {4: 'tense-2'}),
        ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB", '{3} {4} to {5} {1}', {4: 'tense-2'}),
        ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB $VP", '{3} {4} to {5} {1} {6}', {4: 'tense-2'}),
        ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $NP $IN $VP", '{3} {4} {5} {6} {1} {7}', {4: 'tense-2'}),
        ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP/$S/$VP/$SBAR/$SQ", '{3} {4} {1} {5}', {4: 'tense-2'}),
        ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $PP $PP/$S/$VP/$SBAR", '{3} {4} {1} {5} {6}', {4: 'tense-2'}),

        # What/who/how $Be
        # Watch out for things that end in a preposition
        ConstituencyRule("$WHP:what/which/who $Be/$MD $NP of $NP $Verb/$Part $IN", '{3} of {4} {2} {5} {6} {1}'),
        ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $NP $IN", '{3} {2} {4} {5} {1}'),
        ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $VP/$IN", '{3} {2} {4} {1}'),
        ConstituencyRule("$WHP:what/which/who $Be/$MD $NP $IN $NP/$VP", '{1} {2} {3} {4} {5}'),
        ConstituencyRule('$WHP:what/which/who $Be/$MD $NP $Verb $PP', '{3} {2} {4} {1} {5}'),
        ConstituencyRule('$WHP:what/which/who $Be/$MD $NP/$VP/$PP', '{1} {2} {3}'),
        ConstituencyRule("$WHP:how $Be/$MD $NP $VP", '{3} {2} {4} by {1}'),

        # What/who $Verb
        ConstituencyRule("$WHP:what/which/who $VP", '{1} {2}'),

        # $IN what/which $NP
        ConstituencyRule('$IN what/which $NP $Do $NP $Verb $NP', '{5} {6} {7} {1} the {3} of {0}',
                                         {1: 'lower', 6: 'tense-4'}),
        ConstituencyRule('$IN what/which $NP $Be $NP $VP/$ADJP', '{5} {4} {6} {1} the {3} of {0}',
                                         {1: 'lower'}),
        ConstituencyRule('$IN what/which $NP $Verb $NP/$ADJP $VP', '{5} {4} {6} {1} the {3} of {0}',
                                         {1: 'lower'}),
        FindWHPRule(),
]

# Rules for going from WHP to an answer constituent
WHP_RULES = [
        # WHPP rules
        ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun', '{1} {0} {4}'),
        ConstituencyRule('$IN what/which type/sort/kind/group of $NP/$Noun $PP', '{1} {0} {4} {5}'),
        ConstituencyRule('$IN what/which $NP', '{1} the {3} of {0}'),
        ConstituencyRule('$IN $WP/$WDT', '{1} {0}'),

        # what/which
        ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun', '{0} {3}'),
        ConstituencyRule('what/which type/sort/kind/group of $NP/$Noun $PP', '{0} {3} {4}'),
        ConstituencyRule('what/which $NP', 'the {2} of {0}'),

        # How many
        ConstituencyRule('how many/much $NP', '{0} {2}'),

        # Replace
        ReplaceRule('what'),
        ReplaceRule('who'),
        ReplaceRule('how many'),
        ReplaceRule('how much'),
        ReplaceRule('which'),
        ReplaceRule('where'),
        ReplaceRule('when'),
        ReplaceRule('why'),
        ReplaceRule('how'),

        # Just give the answer
        AnswerRule(),
]
