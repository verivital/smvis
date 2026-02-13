#!/usr/bin/env python3
"""Cross-platform setup script for smvis.

Usage:
    python setup_env.py

Creates a virtual environment, installs dependencies, and verifies the setup.
"""
import os
import subprocess
import sys
import shutil


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(root, ".venv")

    # Detect platform
    is_windows = sys.platform == "win32"
    if is_windows:
        python = os.path.join(venv_dir, "Scripts", "python.exe")
        pip = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python = os.path.join(venv_dir, "bin", "python")
        pip = os.path.join(venv_dir, "bin", "pip")

    # Step 1: Create venv
    if not os.path.exists(venv_dir):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print(f"  Created: {venv_dir}")
    else:
        print(f"Virtual environment already exists: {venv_dir}")

    # Step 2: Upgrade pip
    print("\nUpgrading pip...")
    subprocess.check_call([python, "-m", "pip", "install", "--upgrade", "pip"],
                          stdout=subprocess.DEVNULL)

    # Step 3: Install package in editable mode with dev dependencies
    print("Installing smvis with dev dependencies...")
    subprocess.check_call([pip, "install", "-e", f"{root}[dev]"])

    # Step 4: Verify dd.autoref imports
    print("\nVerifying dd.autoref...")
    result = subprocess.run(
        [python, "-c", "import dd.autoref; print('  dd.autoref OK')"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("  WARNING: dd.autoref import failed!")
        print(f"  {result.stderr.strip()}")
        print("  Try: pip install dd --install-option='--fetch'")
    else:
        print(result.stdout.strip())

    # Step 5: Smoke test -- parse a model
    print("\nRunning smoke test (parse counter.smv)...")
    smoke_test = """
import sys
sys.path.insert(0, 'src')
from smvis.smv_parser import parse_smv_file
model = parse_smv_file('examples/counter.smv')
print(f'  Parsed {len(model.variables)} variables, {len(model.specs)} specs')
from smvis.explicit_engine import explore
result = explore(model)
print(f'  Reachable: {len(result.reachable_states)} / {result.total_states} states')
"""
    result = subprocess.run(
        [python, "-c", smoke_test],
        capture_output=True, text=True, cwd=root,
    )
    if result.returncode != 0:
        print("  Smoke test FAILED:")
        print(f"  {result.stderr.strip()}")
        return 1
    else:
        print(result.stdout.strip())

    # Step 6: Success
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    if is_windows:
        activate = f".venv\\Scripts\\activate"
    else:
        activate = "source .venv/bin/activate"
    print(f"\nTo activate:  {activate}")
    print(f"To launch:    python -m smvis")
    print(f"To test:      pytest")
    print(f"Web UI:       http://localhost:8050")
    return 0


if __name__ == "__main__":
    sys.exit(main())
