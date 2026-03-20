from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app


if __name__ == '__main__':
    app = create_app()
    gallery_index = app.extensions['gallery_index']
    summary = gallery_index.rebuild()
    print('Rebuild complete')
    for key, value in summary.items():
        print(f'{key}: {value}')
