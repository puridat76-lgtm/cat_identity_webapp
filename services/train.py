from __future__ import annotations

import copy
import threading
import time
import uuid
from typing import Callable


class TrainManager:
    def __init__(self, rebuild_fn: Callable, summary_fn: Callable):
        self.rebuild_fn = rebuild_fn
        self.summary_fn = summary_fn
        self._lock = threading.Lock()
        self._job = self._empty_job()

    def snapshot(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._job)

    def start(self) -> tuple[dict, bool]:
        with self._lock:
            if self._job['status'] == 'running':
                return copy.deepcopy(self._job), False

            job = self._empty_job()
            job['job_id'] = uuid.uuid4().hex
            job['status'] = 'running'
            job['started_at'] = time.time()
            self._job = job
            job_id = job['job_id']

        thread = threading.Thread(target=self._run, args=(job_id,), daemon=True)
        thread.start()
        return self.snapshot(), True

    def _run(self, job_id: str) -> None:
        started_monotonic = time.monotonic()

        def on_progress(payload: dict) -> None:
            with self._lock:
                if self._job['job_id'] != job_id:
                    return

                elapsed = round(time.monotonic() - started_monotonic, 2)
                total_images = int(payload.get('total_images', self._job['total_images']))
                processed_images = int(payload.get('processed_images', self._job['processed_images']))
                valid_images = int(payload.get('valid_images', self._job['valid_images']))
                split_totals = payload.get('split_totals', self._job['split_totals'])
                split_processed = payload.get('split_processed', self._job['split_processed'])
                progress = round(processed_images / total_images, 4) if total_images else 1.0

                self._job.update(
                    {
                        'status': 'running',
                        'stage': payload.get('stage', self._job['stage']),
                        'total_images': total_images,
                        'processed_images': processed_images,
                        'valid_images': valid_images,
                        'split_totals': split_totals,
                        'split_processed': split_processed,
                        'current_label': payload.get('current_label'),
                        'current_image': payload.get('current_image'),
                        'elapsed_seconds': elapsed,
                        'progress': progress,
                    }
                )
                self._job['history'].append(
                    {
                        'elapsed_seconds': elapsed,
                        'processed_images': processed_images,
                        'valid_images': valid_images,
                        'progress': progress,
                        'gallery_images': split_processed.get('gallery', 0),
                        'not_cat_images': split_processed.get('not_cat', 0),
                        'unknown_cat_images': split_processed.get('unknown_cat', 0),
                    }
                )

        try:
            summary = self.rebuild_fn(progress_callback=on_progress)
            with self._lock:
                if self._job['job_id'] != job_id:
                    return
                self._job.update(
                    {
                        'status': 'completed',
                        'stage': 'completed',
                        'finished_at': time.time(),
                        'elapsed_seconds': round(time.monotonic() - started_monotonic, 2),
                        'progress': 1.0,
                        'summary': {**self.summary_fn(), **summary},
                    }
                )
        except Exception as exc:  # pragma: no cover
            with self._lock:
                if self._job['job_id'] != job_id:
                    return
                self._job.update(
                    {
                        'status': 'failed',
                        'stage': 'failed',
                        'finished_at': time.time(),
                        'elapsed_seconds': round(time.monotonic() - started_monotonic, 2),
                        'error': str(exc),
                    }
                )

    @staticmethod
    def _empty_job() -> dict:
        return {
            'job_id': None,
            'status': 'idle',
            'stage': 'idle',
            'progress': 0.0,
            'started_at': None,
            'finished_at': None,
            'elapsed_seconds': 0.0,
            'total_images': 0,
            'processed_images': 0,
            'valid_images': 0,
            'current_label': None,
            'current_image': None,
            'split_totals': {'gallery': 0, 'not_cat': 0, 'unknown_cat': 0},
            'split_processed': {'gallery': 0, 'not_cat': 0, 'unknown_cat': 0},
            'history': [],
            'summary': {},
            'error': None,
        }
