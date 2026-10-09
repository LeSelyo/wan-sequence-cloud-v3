"""The image runs Python 3.10 (Ubuntu 22.04), the PC runs a newer one: a script written with Python 3.12 syntax parses on the PC and breaks in the image.

It already happened once: `{len(re.findall(r'\\s', text))}` inside an f-string (a backslash in an f-string expression) is valid from Python 3.12 only, and `story_writer.py` could
not be imported in the image. On Python >= 3.12 the tokenizer shows the expression parts of f-strings, so the two 3.12-only forms are searched for directly; on an older Python the
files are simply parsed (an old parser rejects them by itself).
"""
import ast
import io
import sys
import tokenize
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted([*ROOT.glob("scripts/**/*.py"), *ROOT.glob("app/**/*.py")])


def python312_fstring_forms(text: str) -> list[tuple[int, str]]:
    """(line, what) of every f-string expression that only Python >= 3.12 accepts: a backslash inside it, or a quote of the same kind as the f-string's own."""
    start, end = getattr(tokenize, "FSTRING_START", None), getattr(tokenize, "FSTRING_END", None)
    if start is None:  # Python < 3.12: the parser itself refuses these forms
        return []
    found, open_quotes = [], []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == start:
            open_quotes.append(token.string.lstrip("rRbBuUfF"))
        elif token.type == end:
            open_quotes.pop()
        elif open_quotes and token.type == tokenize.STRING:  # a string literal inside an expression part
            if "\\" in token.string:
                found.append((token.start[0], "a backslash inside an f-string expression"))
            quote = open_quotes[-1]
            if len(quote) == 1 and token.string.lstrip("rRbBuUfF").startswith(quote):
                found.append((token.start[0], f"the quote {quote} reused inside an f-string delimited by it"))
    return found


def test_the_checker_finds_the_form_that_broke_the_image():
    broken = "import re\ntext = 'a b'\nline = f\"({len(re.findall(r'[.!?]+(?:\\s|$)', text))} sentences)\"\n"
    if sys.version_info < (3, 12):
        pytest.skip("this Python refuses the form by itself")
    assert python312_fstring_forms(broken) == [(3, "a backslash inside an f-string expression")]
    assert python312_fstring_forms("d = {'k': 1}\nline = f\"{d['k']}\"\n") == []
    assert python312_fstring_forms("d = {'k': 1}\nline = f\"{d[\"k\"]}\"\n") == [(2, 'the quote " reused inside an f-string delimited by it')]
    assert python312_fstring_forms("d = {'k': 1}\nline = f'''{d[\"k\"]}'''\n") == []  # a triple-quoted f-string may hold the single quote


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_script_of_the_image_is_valid_for_python_310(path):
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    assert python312_fstring_forms(text) == [], "valid only from Python 3.12, but the image runs Python 3.10"
