"""Utilities for AddSent generator."""

class ConstituencyParse(object):
    """A CoreNLP constituency parse (or a node in a parse tree).

    Word-level constituents have |word| and |index| set and no children.
    Phrase-level constituents have no |word| or |index| and have at least one child.
    """
    def __init__(self, tag, children=None, word=None, index=None):
        self.tag = tag
        if children:
            self.children = children
        else:
            self.children = None
        self.word = word
        self.index = index

    @classmethod
    def _recursive_parse_corenlp(cls, tokens, i, j):
        orig_i = i
        if tokens[i] == '(':
            tag = tokens[i + 1]
            children = []
            i = i + 2
            while True:
                child, i, j = cls._recursive_parse_corenlp(tokens, i, j)
                if isinstance(child, cls):
                    children.append(child)
                    if tokens[i] == ')': 
                        return cls(tag, children), i + 1, j
                else:
                    if tokens[i] != ')':
                        raise ValueError('Expected ")" following leaf')
                    return cls(tag, word=child, index=j), i + 1, j + 1
        else:
            # Only other possibility is it's a word
            return tokens[i], i + 1, j

    @classmethod
    def from_corenlp(cls, s):
        """Parses the "parse" attribute returned by CoreNLP parse annotator."""
        # "parse": "(ROOT\n  (SBARQ\n    (WHNP (WDT What)\n      (NP (NN portion)\n        (PP (IN                       of)\n          (NP\n            (NP (NNS households))\n            (PP (IN in)\n              (NP (NNP             Jacksonville)))))))\n    (SQ\n      (VP (VBP have)\n        (NP (RB only) (CD one) (NN person))))\n    (. ?        )))",
        s_spaced = s.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
        tokens = [t for t in s_spaced.split(' ') if t]
        tree, index, num_words = cls._recursive_parse_corenlp(tokens, 0, 0)
        if index != len(tokens):
            raise ValueError('Only parsed %d of %d tokens' % (index, len(tokens)))
        return tree

    def is_singleton(self):
        if self.word:
            return True
        if len(self.children) > 1:
            return False
        return self.children[0].is_singleton()
        
    def print_tree(self, indent=0):
        spaces = '  ' * indent
        if self.word:
            print(f"{spaces}{self.tag}: {self.word} ({self.index})")
        else:
            print(f"{spaces}{self.tag}")
            for c in self.children:
                c.print_tree(indent=indent + 1)

    def get_phrase(self):
        if self.word:
            return self.word
        toks = []
        for i, c in enumerate(self.children):
            p = c.get_phrase()
            if i == 0 or p.startswith("'"):
                toks.append(p)
            else:
                toks.append(' ' + p)
        return ''.join(toks)

    def get_start_index(self):
        if self.index is not None:
            return self.index
        return self.children[0].get_start_index()

    def get_end_index(self):
        if self.index is not None:
            return self.index + 1
        return self.children[-1].get_end_index()

    @classmethod
    def _recursive_replace_words(cls, tree, new_words, i):
        if tree.word:
            new_word = new_words[i]
            return (cls(tree.tag, word=new_word, index=tree.index), i + 1)
        new_children = []
        for c in tree.children:
            new_child, i = cls._recursive_replace_words(c, new_words, i)
            new_children.append(new_child)
        return cls(tree.tag, children=new_children), i

    @classmethod
    def replace_words(cls, tree, new_words):
        """Return a new tree, with new words replacing old ones."""
        new_tree, i = cls._recursive_replace_words(tree, new_words, 0)
        if i != len(new_words):
            raise ValueError('len(new_words) == %d != i == %d' % (len(new_words), i))
        return new_tree

def rejoin(tokens, sep=None):
    """Rejoin tokens into the original sentence.

    Args:
    tokens: a list of dicts containing 'originalText' and 'before' fields.
        All other fields will be ignored.
    sep: if provided, use the given character as a separator instead of
        the 'before' field (e.g. if you want to preserve where tokens are).
    Returns: the original sentence that generated this CoreNLP token list.
    """
    if sep is None:
        return ''.join('%s%s' % (t['before'], t['originalText']) for t in tokens)
    else:
        # Use the given separator instead
        return sep.join(t['originalText'] for t in tokens)


def get_tokens_for_answers(answer_objs, corenlp_obj):
    """Get CoreNLP tokens corresponding to a SQuAD answer object."""
    first_a_toks = None
    for i, a_obj in enumerate(answer_objs):
        a_toks = []
        answer_start = a_obj['answer_start']
        answer_end = answer_start + len(a_obj['text'])
        for sent in corenlp_obj['sentences']:
            for tok in sent['tokens']:
                if tok['characterOffsetBegin'] >= answer_end:
                    continue
                if tok['characterOffsetEnd'] <= answer_start:
                    continue
                a_toks.append(tok)
        if rejoin(a_toks).strip() == a_obj['text']:
            # Make sure that the tokens reconstruct the answer
            return i, a_toks
        if i == 0:
            first_a_toks = a_toks
    # None of the extracted token lists reconstruct the answer
    # Default to the first
    return 0, first_a_toks

def get_determiner_for_answers(answer_objs):
    for ans in answer_objs:
        words = ans['text'].split(' ')
        if words[0].lower() == 'the':
            return 'the'
        if words[0].lower() in ('a', 'an'):
            return 'a'
    return None

def compress_whnp(tree, inside_whnp=False):
    if not tree.children: return tree  # Reached leaf
    # Compress all children
    for i, c in enumerate(tree.children):
        tree.children[i] = compress_whnp(c, inside_whnp=inside_whnp or tree.tag == 'WHNP')
    if tree.tag != 'WHNP':
        if inside_whnp:
            # Wrap everything in an NP
            return ConstituencyParse('NP', children=[tree])
        return tree
    wh_word = None
    new_np_children = []
    new_siblings = []
    for i, c in enumerate(tree.children):
        if i == 0:
            if c.tag in ('WHNP', 'WHADJP', 'WHAVP', 'WHPP'):
                wh_word = c.children[0]
                new_np_children.extend(c.children[1:])
            elif c.tag in ('WDT', 'WP', 'WP$', 'WRB'):
                wh_word = c
            else:
                # No WH-word at start of WHNP
                return tree
        else:
            if c.tag == 'SQ':  # Due to bad parse, SQ may show up here
                new_siblings = tree.children[i:]
                break
            # Wrap everything in an NP
            new_np_children.append(ConstituencyParse('NP', children=[c]))
    if new_np_children:
        new_np = ConstituencyParse('NP', children=new_np_children)
        new_tree = ConstituencyParse('WHNP', children=[wh_word, new_np])
    else:
        new_tree = tree
    if new_siblings:
        new_tree = ConstituencyParse('SBARQ', children=[new_tree] + new_siblings)
    return new_tree

def read_const_parse(parse_str):
    tree = ConstituencyParse.from_corenlp(parse_str)
    new_tree = compress_whnp(tree)
    return new_tree
