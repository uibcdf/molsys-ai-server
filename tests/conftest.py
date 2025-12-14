import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT / "client"
SERVER = ROOT / "server"

for path in (str(CLIENT), str(SERVER)):
    if path not in sys.path:
        sys.path.insert(0, path)

