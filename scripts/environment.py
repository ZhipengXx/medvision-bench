"""Record software versions and generic hardware without host or process details."""
import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.common.runtime import ROOT, save_json

PACKAGES = ['torch', 'torchvision', 'numpy', 'scipy', 'scikit-learn', 'h5py', 'nibabel',
            'MedPy', 'matplotlib', 'Pillow', 'PyYAML', 'PyMuPDF', 'pytest', 'requests']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/environment.json')
    parser.add_argument('--write-lock', action='store_true',
                        help='Explicitly replace the software lock with installed package versions')
    args = parser.parse_args()
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    record = dict(python=platform.python_version(), packages=versions,
                  cuda=torch.version.cuda,
                  training_gpu_model=torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                  scope='Software versions and generic hardware only; host topology and process details are omitted.')
    save_json(args.output, record)
    if args.write_lock:
        (ROOT / 'requirements.lock.txt').write_text(
            '\n'.join(f'{name}=={version}' for name, version in versions.items()) + '\n')
    print('Recorded software versions and generic hardware')


if __name__ == '__main__':
    main()
