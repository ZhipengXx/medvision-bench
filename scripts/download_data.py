"""Download ISIC 2018 Task 3 files from the public challenge archive.

Left atrial volumes are acquired separately under the original provider's
access conditions; see docs/data.md. No private distribution link is embedded.
"""
import hashlib
import json
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_PAGE = 'https://challenge.isic-archive.com/data/'


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    partial = destination.with_suffix(destination.suffix + '.partial')
    print('Downloading', destination.name, flush=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for chunk in response.iter_content(2**20):
                stream.write(chunk)
    partial.replace(destination)


def extract_archive(archive):
    marker = archive.with_suffix('.extracted')
    if marker.exists() or not zipfile.is_zipfile(archive):
        return
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            target = (archive.parent / info.filename).resolve()
            if not target.is_relative_to(archive.parent.resolve()):
                raise ValueError('Unsafe archive path')
        handle.extractall(archive.parent)
    marker.touch()


def main():
    sources = ROOT / 'results/sources'
    sources.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_PAGE, timeout=60)
    response.raise_for_status()
    links = re.findall(r'href=[\"\']([^\"\']+)[\"\']', response.text)
    urls = sorted({urljoin(DATA_PAGE, link) for link in links
                   if 'ISIC2018_Task3' in link and ('.zip' in link or '.csv' in link)})
    if not urls:
        raise RuntimeError('No Task 3 archives found; check the current challenge data page')
    manifest = []
    for url in urls:
        destination = ROOT / 'data/isic' / Path(urlparse(url).path).name
        fetch(url, destination)
        with destination.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest.append(dict(url=url, file=destination.relative_to(ROOT).as_posix(), sha256=digest))
        extract_archive(destination)
    (sources / 'isic_downloads.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
