#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run from repo root; optionally pass file path. Example: python scripts/check_length.py path/to/file.md"""

import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

path = sys.argv[1] if len(sys.argv) > 1 else 'Response_to_Reviewer_1_OpenReview_Short.md'
if not os.path.isfile(path):
    print(f"File not found: {path}")
    sys.exit(1)

with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

char_count = len(content)
char_count_no_spaces = len(content.replace(' ', ''))
line_count = len(content.splitlines())
word_count = len(content.split())

print(f'Character count (total): {char_count}')
print(f'Character count (no spaces): {char_count_no_spaces}')
print(f'Line count: {line_count}')
print(f'Word count: {word_count}')
print(f'\nLimit: 5000 characters')
print(f'Remaining: {5000 - char_count} characters')
print(f'Percentage used: {char_count/5000*100:.1f}%')
