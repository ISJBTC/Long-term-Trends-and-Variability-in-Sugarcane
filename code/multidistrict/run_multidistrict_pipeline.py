"""Run multidistrict data preparation and figure generation."""
import subprocess
import sys
import os

base = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(base))

print("Step 1: Data preparation...")
r1 = subprocess.run([sys.executable, os.path.join(base, 'multidistrict_data_preparation.py')])
if r1.returncode != 0:
    sys.exit(1)

print("\nStep 2: Create all 12 figures...")
r2 = subprocess.run([sys.executable, os.path.join(base, 'multidistrict_create_all_figures.py')])
if r2.returncode != 0:
    sys.exit(1)

print("\nPipeline complete.")
