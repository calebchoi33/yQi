#!/usr/bin/env python3
"""Create semantic vector database - convenience wrapper."""

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    script_path = Path(__file__).parent / 'src' / 'scripts' / 'create_semantic_vdb.py'
    subprocess.run([sys.executable, str(script_path)] + sys.argv[1:])
