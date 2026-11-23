"""One-step setup: python install.py

Creates a virtual environment in .venv, installs Directed Growing Sheet into it
(with Agent Skills support), runs a self-check that needs no API key, and asks
the one thing only you can answer: which model to use (Gemini 4 Hydrogen by
default). Run it again at any time; it only adds what is missing.

python install.py --skip-model installs without asking (for scripts and CI).
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


def python_in_venv() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def run(*command) -> None:
    print("$", " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def main() -> int:
    if sys.version_info < (3, 10):  # noqa: UP036 (install.py may be run by an old Python)
        print("Python 3.10 or later is needed; this is", sys.version.split()[0])
        return 1
    if not python_in_venv().exists():
        print("Creating the virtual environment in .venv ...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    python = python_in_venv()
    run(python, "-m", "pip", "install", "--quiet", "--upgrade", "pip")
    run(python, "-m", "pip", "install", "--quiet", "-e", ".[skills]")
    print("\nSelf-check (no API key needed):")
    run(python, "quickstart.py", "--demo")
    if "--skip-model" in sys.argv or not sys.stdin.isatty():
        print("\nModel not chosen yet: run python quickstart.py --setup later.")
    else:
        subprocess.run([str(python), "quickstart.py", "--setup"], cwd=ROOT, check=False)
    activate = r".venv\Scripts\activate" if sys.platform == "win32" else "source .venv/bin/activate"
    print(
        "\nDone. To chat with the assistant:\n"
        f"  {activate}\n"
        "  python quickstart.py\n"
        "See GETTING_STARTED.md for details."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
