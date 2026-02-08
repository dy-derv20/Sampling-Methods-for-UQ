"""
Utility functions for CAP 6938 Assignment 1: Foundations of Sampling Methods
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Set up paths
FIGS_DIR = Path(__file__).parent.parent / "figs"
DATA_DIR = Path(__file__).parent.parent / "data"

# Ensure directories exist
FIGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def savefig(filename):
    """Save figure to figs directory."""
    filepath = FIGS_DIR / filename
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"Figure saved: {filepath}")
    plt.close()


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


# Command-line interface helpers
_funcs = {}


def handle(number):
    """Decorator to register a function as a question handler."""
    def register(func):
        _funcs[number] = func
        return func
    return register


def run(question):
    """Run a specific question handler."""
    if question not in _funcs:
        raise ValueError(f"Unknown question: {question}")
    return _funcs[question]()


def main():
    """Main entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="CAP 6938 Assignment 1: Foundations of Sampling Methods"
    )
    parser.add_argument(
        "question",
        choices=sorted(_funcs.keys()) + ["all"],
        help="Question number to run, or 'all' to run all questions"
    )
    args = parser.parse_args()

    if args.question == "all":
        for q in sorted(_funcs.keys(), key=lambda x: [int(p) if p.isdigit() else p for p in x.split('.')]):
            start = f"== {q} "
            print("\n" + start + "=" * (80 - len(start)))
            run(q)
    else:
        return run(args.question)
