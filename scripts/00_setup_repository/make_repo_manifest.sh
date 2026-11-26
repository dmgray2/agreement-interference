#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
temp_dir="$(mktemp -d)"
trap 'rm -f "$temp_dir/release.zip"; rmdir "$temp_dir"' EXIT
bash "$root/scripts/00_setup_repository/make_release_zip.sh" --output "$temp_dir/release.zip"
python3 - "$root" "$temp_dir/release.zip" <<'PYCODE'
from pathlib import Path
import sys
from zipfile import ZipFile
root = Path(sys.argv[1])
with ZipFile(sys.argv[2]) as archive:
    for name in ('repo_manifest.txt', 'repo_manifest.csv'):
        (root / name).write_bytes(archive.read(name))
PYCODE
