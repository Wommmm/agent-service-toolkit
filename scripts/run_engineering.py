"""Load an explicitly ignored local config, then use the existing API or UI entrypoint."""
import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=root / ".env.engineering")
    parser.add_argument("--ui", action="store_true")
    args = parser.parse_args()
    if not args.env_file.is_file():
        raise SystemExit("Create a private environment file from .env.example and pass --env-file.")
    load_dotenv(args.env_file, override=False)
    os.chdir(root)
    (root / "data").mkdir(exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    if args.ui:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(root / "src/streamlit_app.py"),
                        "--server.address", "127.0.0.1", "--server.port", "8502"], check=True)
    else:
        runpy.run_path(str(root / "src/run_service.py"), run_name="__main__")


if __name__ == "__main__":
    main()
