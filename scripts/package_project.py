"""Package the public project with a SHA-256 manifest and archive verification."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_project import ROOT, verify_project

EXCLUDED_DIRS = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'data',
                 'dist', 'report_preview', 'release_review'}
EXCLUDED_SUFFIXES = {'.pt', '.pth', '.ckpt', '.npz', '.npy', '.h5', '.hdf5', '.nii',
                     '.gz', '.pyc', '.aux', '.out', '.log', '.zip'}
EXCLUDED_NAMES = {'progress.log', '.env', 'release_manifest.json'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def public_files():
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        licensed_cam_cache = relative.parts[:2] == ('results', 'gradcam_assets') and path.suffix.lower() == '.npz'
        if path.name in EXCLUDED_NAMES or (path.suffix.lower() in EXCLUDED_SUFFIXES and not licensed_cam_cache):
            continue
        if path.name.startswith('.env'):
            continue
        yield path, relative.as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/medvision-bench.zip')
    args = parser.parse_args()
    verification = verify_project()
    entries = list(public_files())
    entries = [(path, name) for path, name in entries if path.resolve() != args.output.resolve()]
    manifest = dict(project='medvision-bench', verification=verification,
                    files=[dict(path=name, bytes=path.stat().st_size, sha256=digest(path))
                           for path, name in entries])
    manifest_text = json.dumps(manifest, indent=2) + '\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, name in entries:
            archive.write(path, 'medvision-bench/' + name)
        archive.writestr('medvision-bench/docs/release_manifest.json', manifest_text)
    with zipfile.ZipFile(args.output) as archive:
        if archive.testzip():
            raise ValueError('Archive CRC verification failed')
        for entry in manifest['files']:
            content = archive.read('medvision-bench/' + entry['path'])
            if hashlib.sha256(content).hexdigest() != entry['sha256']:
                raise ValueError('Archive member SHA-256 verification failed')
    print(f'Packaged {len(entries)} public files ({args.output.stat().st_size / 2**20:.1f} MiB)')
    print('Archive SHA-256:', digest(args.output))


if __name__ == '__main__':
    main()
