

import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
from scripts.parsers import MolmoParser
from scripts.unification_common import run_unification

def main():
    parser = argparse.ArgumentParser(description="Unify Molmo Annotations")
    parser.add_argument("--root", type=str, default="X:\\data\\Schwan_T3_Clean", help="Root directory containing session folders")
    args = parser.parse_args()

    # Setup parsers
    parsers = [
        MolmoParser()
    ]

    # Run
    run_unification(args.root, parsers, output_filename="unified_annotations_molmo.json")

if __name__ == "__main__":
    main()
