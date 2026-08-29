"""Turns text into the bag of content words that both the reward in src/training/metrics.py and
the shortcut gate in src/generation/shortcut.py compare against. It lives at the top level
because generation and training both need the same definition and neither may import the other.
"""

import re

_WORD_PATTERN = re.compile(r"[a-z0-9$%.]+")

# Closed-class words carry no grounding signal, so any fluent text would
# otherwise share them with everything else.
_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his i if in into is it
    its of on or she that the their them there these they this to was were what when which who
    will with would you your""".split()
)


def content_tokens(text: str) -> set[str]:
    """Tokenize text into lowercased content words.

    Params:
        text: Arbitrary text.

    Returns:
        The set of content tokens.
    """
    tokens = (token.strip(".") for token in _WORD_PATTERN.findall(text.lower()))
    return {token for token in tokens if token and token not in _STOPWORDS}
