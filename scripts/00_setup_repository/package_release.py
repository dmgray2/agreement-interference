"""Package the explicitly listed CE-Agree release files."""
import argparse
import csv
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
BLOCKED_DIRS = {'.git', '__pycache__', '.Rproj.user', '.ipynb_checkpoints',
                'eacl paper', 'paper', 'reviews', 'raw_batches',
                'batch_requests', 'batch_manifests'}
BLOCKED_FILE = re.compile(
    r'nonce|405b|(?:^|[/_.-])ce[03](?:pos|(?=[/_.-]|$))|'
    r'api[ _-]*(?:key|[.]txt)|(?:^|/)[.]env(?:$|[.])|'
    r':[A-Za-z]|[.](?:pem|key|docx|zip|aux|log|blg|bbl|out)$', re.I)
SECRET = re.compile(rb'(?:sk-(?:proj-|ant-api[0-9]+-)?[A-Za-z0-9_-]{24,}'
                    rb'|gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_[A-Za-z0-9_]{25,})')

def read_paths():
    listing = ROOT / 'release_files.txt'
    lines = [line for line in listing.read_text(encoding='utf-8').splitlines()
             if line and not line.startswith('#')]
    if not lines or len(lines) != len(set(lines)):
        raise SystemExit('Release list is empty or contains duplicate paths.')
    for name in lines:
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or BLOCKED_DIRS.intersection(part.casefold() for part in relative.parts)
                or BLOCKED_FILE.search(name)):
            raise SystemExit(f'Excluded release path: {name}')
        path = ROOT / relative
        if not path.is_file() or path.resolve() != path:
            raise SystemExit(f'Missing release file or symlink: {name}')
        yield name, path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f'Output already exists: {output}')
    csv_buffer = io.StringIO(newline='')
    writer = csv.writer(csv_buffer, lineterminator='\n')
    writer.writerow(('path', 'bytes', 'sha256'))
    text_manifest = ['SHA256  Bytes  Path']
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary) / output.name
        with ZipFile(candidate, 'w', ZIP_DEFLATED) as archive:
            for name, path in read_paths():
                content = path.read_bytes()
                if path.suffix.lower() not in {'.pdf', '.png', '.jpg', '.gz'} and SECRET.search(content):
                    raise SystemExit(f'Potential credential in {name}')
                digest = hashlib.sha256(content).hexdigest()
                writer.writerow((name, len(content), digest))
                text_manifest.append(f'{digest}  {len(content)}  {name}')
                archive.writestr(name, content)
            archive.writestr('repo_manifest.csv', csv_buffer.getvalue())
            archive.writestr('repo_manifest.txt', '\n'.join(text_manifest) + '\n')
        with ZipFile(candidate) as archive:
            failed = archive.testzip()
            if failed is not None:
                raise SystemExit(f'Invalid archive member: {failed}')
        output.hardlink_to(candidate)
    print(output)

if __name__ == '__main__':
    main()
