#!/usr/bin/env python

from deepfigures.utils.stringmatch import match


def test_match():
    m = match('hello', 'hello')
    assert m.cost == 0
    assert m.start_pos == 0
    assert m.end_pos == 5

    m = match('e', 'hello')
    assert m.cost == 0
    assert m.start_pos == 1
    assert m.end_pos == 2

    m = match('hello', 'e')
    assert m.cost == 4
    assert m.start_pos == 0
    assert m.end_pos == 1

    # Prefer character omissions over character edits in match bounds
    m = match('bab', 'cac')
    assert m.cost == 2
    assert m.start_pos == 1
    assert m.end_pos == 2

    # Select first match in the text in case of ties
    m = match('ab', 'ba')
    assert m.cost == 1
    assert m.start_pos == 0
    assert m.end_pos == 1

    m = match('hello', 'world')
    assert m.cost == 4
    assert m.start_pos == 1
    assert m.end_pos == 2


def test_unicode_match():
    m = match('æther', 'aether')
    assert m.cost == 1
    assert m.start_pos == 2
    assert m.end_pos == 6

    m = match('こんにちは世界', 'こんばんは世界')
    assert m.cost == 2
    assert m.start_pos == 0
    assert m.end_pos == 7


if __name__ == '__main__':
    import pytest

    pytest.main([__file__])
