#!/usr/bin/env python3
"""Command-line evaluation runner for RAG system - convenience wrapper."""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / 'scripts' / 'run_eval.py'
    subprocess.run([sys.executable, str(script_path)] + sys.argv[1:])
