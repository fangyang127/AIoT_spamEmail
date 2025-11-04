"""Pytest configuration helpers.

Ensure the repository root is on sys.path so tests can import `src` modules
when pytest is invoked from different working directories or environments.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
