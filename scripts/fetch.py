#!/usr/bin/env python3
"""Download files listed in a plain-text URL list into data/raw/<layer>/.

Usage:  python scripts/fetch.py data/urls.txt [--dry-run]
Each line:  <layer_id> <url>        (blank lines and # comments ignored)

Run this on a machine that can reach NASA/USGS/PDS hosts. It was NOT tested against
the real servers (the build sandbox blocked them). Uses only the standard library.
"""
import sys, pathlib, urllib.request, urllib.parse

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    if not args:
        sys.exit(__doc__)
    root = pathlib.Path(__file__).resolve().parent.parent / "data" / "raw"
    for line in pathlib.Path(args[0]).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        layer, url = line.split(None, 1)
        name = pathlib.Path(urllib.parse.urlparse(url).path).name or "download"
        dest = root / layer / name
        print(("[dry-run] " if dry else "") + f"{url} -> {dest}")
        if dry or dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)

if __name__ == "__main__":
    main()
