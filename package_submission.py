import hashlib
import json
from pathlib import Path


def write_manifest(root: Path):
    files = sorted(p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts)
    manifest = {}
    for path in files:
        rel = path.relative_to(root).as_posix()
        manifest[rel] = {
            'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (root / 'submission_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Manifest saved to submission_manifest.json')


if __name__ == '__main__':
    write_manifest(Path(__file__).resolve().parent)
