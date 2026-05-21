"""
inventropy - A Python package for computing inverse entropy in language models

This package provides a simple interface to calculate inverse entropy,
a metric for evaluating language model consistency and reliability.
"""

__version__ = "0.1.0"
__author__ = "Haoyi Song"
__email__ = "haoyiso@umich.edu"

from .core import calculate_inv_entropy

__all__ = ["calculate_inv_entropy"]
