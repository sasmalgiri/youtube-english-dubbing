"""GPU Pool Manager — distribute dubbing work across multiple free GPU backends.

Each backend runs the VoiceDub English backend (app.py) on a free GPU
(Colab T4, Kaggle T4, Lightning AI, local NVIDIA, etc.).

The pool submits the SAME video URL to each backend, but each backend
processes it with split_duration so only one part is dubbed per backend.

Usage:
    pool = GPUPool()
    pool.add_backend("http://colab-ngrok-url.ngrok.io")
    pool.add_backend("http://kaggle-ngrok-url.ngrok.io")
    pool.dub_video("https://youtube.com/watch?v=...")
"""

import os
import sys
import time
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests as _requests
except ImportError:
    _requests = None


def _require_requests():
    if _requests is None:
        print("Error: requests library required. Install: pip install requests")
        sys.exit(1)


@dataclass
class GPUBackend:
    """A single GPU backend (Colab, Kaggle, local, etc.)."""
    url: str
    name: str = ""
    gpu_type: str = ""
    vram_gb: float = 0
    is_healthy: bool = False
    active_jobs: int = 0
    total_completed: int = 0
    last_health_check: float = 0

    def health_check(self) -> bool:
        """Ping the backend to check if it's alive."""
        _require_requests()
        try:
            resp = _requests.get(f"{self.url}/api/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self.is_healthy = True
                self.gpu_type = data.get("gpu", "unknown")
                self.vram_gb = data.get("vram_gb", 0)
                self.name = self.name or data.get("hostname", self.url)
                self.last_health_check = time.time()
                return True
        except _requests.ConnectionError:
            pass
        except _requests.Timeout:
            pass
        except Exception:
            pass
        self.is_healthy = False
        return False


@dataclass
class PoolJob:
    """Tracks a job on a backend."""
    backend: GPUBackend
    job_id: str = ""
    state: str = "pending"  # pending, running, done, error
    progress: float = 0.0
    error: str = ""


class GPUPool:
    """Manages multiple GPU backends and distributes dubbing jobs."""

    def __init__(self):
        self.backends: List[GPUBackend] = []
        self._lock = threading.Lock()

    def add_backend(self, url: str, name: str = "") -> GPUBackend:
        """Register a GPU backend."""
        _require_requests()
        url = url.rstrip("/")
        backend = GPUBackend(url=url, name=name or url)
        backend.health_check()
        with self._lock:
            self.backends.append(backend)
        status = "ONLINE" if backend.is_healthy else "OFFLINE"
        gpu_info = f" ({backend.gpu_type}, {backend.vram_gb:.1f}GB)" if backend.is_healthy else ""
        print(f"[Pool] Added {backend.name}: {status}{gpu_info}")
        return backend

    def remove_backend(self, url: str):
        """Remove a backend."""
        url = url.rstrip("/")
        with self._lock:
            self.backends = [b for b in self.backends if b.url != url]

    def get_healthy_backends(self) -> List[GPUBackend]:
        """Return all healthy backends, refreshing stale checks."""
        now = time.time()
        with self._lock:
            for b in self.backends:
                if now - b.last_health_check > 30:
                    b.health_check()
            return [b for b in self.backends if b.is_healthy]

    def _pick_backend(self) -> Optional[GPUBackend]:
        """Pick the least-loaded healthy backend."""
        with self._lock:
            healthy = [b for b in self.backends if b.is_healthy]
            if not healthy:
                return None
            healthy.sort(key=lambda b: (b.active_jobs, -b.vram_gb))
            return healthy[0]

    def _submit_job(self, backend: GPUBackend, video_url: str,
                    settings: Dict) -> PoolJob:
        """Submit a dubbing job to a backend via POST /api/jobs."""
        job = PoolJob(backend=backend)
        try:
            payload = {
                "url": video_url,
                "source_language": settings.get("source_language", "auto"),
                "target_language": settings.get("target_language", "en"),
                "asr_model": settings.get("asr_model", "large-v3-turbo"),
                "translation_engine": settings.get("translation_engine", "auto"),
                "tts_rate": settings.get("tts_rate", "+0%"),
                "mix_original": settings.get("mix_original", False),
                "original_volume": settings.get("original_volume", 0.10),
                "use_cosyvoice": settings.get("use_cosyvoice", True),
                "use_chatterbox": settings.get("use_chatterbox", False),
                "use_elevenlabs": settings.get("use_elevenlabs", False),
                "use_google_tts": settings.get("use_google_tts", False),
                "use_coqui_xtts": settings.get("use_coqui_xtts", False),
                "use_edge_tts": settings.get("use_edge_tts", True),
                "prefer_youtube_subs": settings.get("prefer_youtube_subs", False),
                "use_yt_translate": settings.get("use_yt_translate", False),
                "multi_speaker": settings.get("multi_speaker", False),
                "transcribe_only": settings.get("transcribe_only", False),
                "audio_priority": settings.get("audio_priority", True),
                "audio_bitrate": settings.get("audio_bitrate", "320k"),
                "encode_preset": settings.get("encode_preset", "medium"),
                "split_duration": settings.get("split_duration", 0),
                "fast_assemble": settings.get("fast_assemble", False),
                "enable_manual_review": settings.get("enable_manual_review", True),
            }

            resp = _requests.post(f"{backend.url}/api/jobs", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            job.job_id = data.get("id", "")
            job.state = "running"
            with self._lock:
                backend.active_jobs += 1
            print(f"[Pool] Job {job.job_id} → {backend.name}")

        except Exception as e:
            job.state = "error"
            job.error = str(e)
            print(f"[Pool] FAILED to submit to {backend.name}: {e}")

        return job

    def _poll_job(self, job: PoolJob, timeout: int = 7200) -> PoolJob:
        """Poll GET /api/jobs/{id} until job completes or times out."""
        if job.state == "error":
            return job

        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = _requests.get(
                    f"{job.backend.url}/api/jobs/{job.job_id}",
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    job.state = data.get("state", "running")
                    job.progress = data.get("overall_progress", 0)

                    if job.state == "done":
                        with self._lock:
                            job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
                            job.backend.total_completed += 1
                        print(f"[Pool] Job {job.job_id} DONE on {job.backend.name}")
                        return job

                    if job.state == "error":
                        job.error = data.get("message", "Unknown error")
                        with self._lock:
                            job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
                        print(f"[Pool] Job {job.job_id} ERROR: {job.error}")
                        return job

                    # Print progress
                    pct = int(job.progress * 100)
                    step = data.get("current_step", "")
                    msg = data.get("message", "")
                    print(f"  [{job.backend.name}] {pct}% — {step}: {msg}", end="\r")

            except Exception:
                if not job.backend.health_check():
                    job.state = "error"
                    job.error = "Backend went offline"
                    with self._lock:
                        job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
                    return job

            time.sleep(10)

        job.state = "error"
        job.error = "Timeout"
        with self._lock:
            job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
        return job

    def _download_result(self, job: PoolJob, output_dir: Path, filename: str = "") -> Optional[Path]:
        """Download dubbed video via GET /api/jobs/{id}/result."""
        if job.state != "done":
            return None
        try:
            resp = _requests.get(
                f"{job.backend.url}/api/jobs/{job.job_id}/result",
                stream=True, timeout=300
            )
            resp.raise_for_status()
            fname = filename or f"dubbed_{job.job_id}.mp4"
            out_path = output_dir / fname
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[Pool] Downloaded {fname} ({size_mb:.1f} MB)")
            return out_path
        except Exception as e:
            print(f"[Pool] Download failed for {job.job_id}: {e}")
            return None

    def dub_video(self, video_url: str, settings: Optional[Dict] = None,
                  output_dir: str = "gpu_pool_output") -> Optional[Path]:
        """Dub a video using the best available backend.

        For a single video, picks the best backend and submits.
        For parallel processing of a long video, use dub_parallel().
        """
        _require_requests()
        settings = settings or {}
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        backend = self._pick_backend()
        if not backend:
            print("[Pool] No healthy backends available!")
            return None

        print(f"\n[Pool] Submitting to {backend.name} ({backend.gpu_type})...")
        job = self._submit_job(backend, video_url, settings)
        if job.state == "error":
            return None

        print(f"[Pool] Waiting for job {job.job_id}...")
        job = self._poll_job(job)

        if job.state == "done":
            return self._download_result(job, out_path)
        else:
            print(f"[Pool] Job failed: {job.error}")
            return None

    def dub_parallel(self, video_url: str, settings: Optional[Dict] = None,
                     output_dir: str = "gpu_pool_output") -> List[Path]:
        """Dub a long video by sending to ALL backends simultaneously.

        Each backend gets the same URL with split_duration enabled.
        Parts are downloaded and concatenated locally.

        Strategy: Submit the same job to multiple backends, each with
        split_duration=30. The backends handle splitting internally.
        We collect all results.
        """
        _require_requests()
        settings = settings or {}
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        healthy = self.get_healthy_backends()
        if not healthy:
            print("[Pool] No healthy backends available!")
            return []

        print(f"\n{'='*60}")
        print(f"GPU POOL: {len(healthy)} backends")
        for b in healthy:
            print(f"  - {b.name}: {b.gpu_type} ({b.vram_gb:.1f}GB)")
        print(f"{'='*60}\n")

        # Enable splitting on the backend side
        settings.setdefault("split_duration", 30)

        # Submit to ALL backends — each processes the full video with splitting
        # The first one to complete wins; others provide redundancy
        results = []
        with ThreadPoolExecutor(max_workers=len(healthy)) as pool:
            futures = {}
            for backend in healthy:
                fut = pool.submit(self._submit_and_poll, backend, video_url, settings)
                futures[fut] = backend

            for fut in as_completed(futures):
                backend = futures[fut]
                job = fut.result()
                if job and job.state == "done":
                    path = self._download_result(job, out_path, f"dubbed_{backend.name}.mp4")
                    if path:
                        results.append(path)
                        # First successful result — cancel waiting for others
                        print(f"\n[Pool] Got result from {backend.name}!")
                        break

        if results:
            print(f"\n[Pool] SUCCESS — output: {results[0]}")
        else:
            print(f"\n[Pool] ALL backends failed!")

        return results

    def _submit_and_poll(self, backend, video_url, settings):
        """Submit and wait for completion on a single backend."""
        job = self._submit_job(backend, video_url, settings)
        if job.state == "running":
            job = self._poll_job(job)
        return job

    def status(self) -> Dict:
        """Get pool status."""
        healthy = self.get_healthy_backends()
        return {
            "total_backends": len(self.backends),
            "healthy_backends": len(healthy),
            "total_vram_gb": sum(b.vram_gb for b in healthy),
            "active_jobs": sum(b.active_jobs for b in healthy),
            "total_completed": sum(b.total_completed for b in self.backends),
            "backends": [
                {
                    "name": b.name, "url": b.url, "gpu": b.gpu_type,
                    "vram_gb": b.vram_gb, "healthy": b.is_healthy,
                    "active_jobs": b.active_jobs, "completed": b.total_completed,
                }
                for b in self.backends
            ],
        }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _require_requests()

    print("GPU Pool Manager — Zero-Spend English Dubbing")
    print("=" * 50)

    pool = GPUPool()

    # Auto-detect local backend
    try:
        resp = _requests.get("http://localhost:8000/api/health", timeout=3)
        if resp.status_code == 200:
            pool.add_backend("http://localhost:8000", "Local")
    except Exception:
        pass

    print("\nCommands:")
    print("  add <url> [name]  — Add a GPU backend")
    print("  remove <url>      — Remove a backend")
    print("  list              — Show all backends")
    print("  dub <youtube-url> — Dub on best backend")
    print("  parallel <url>    — Dub on ALL backends (redundancy)")
    print("  quit              — Exit")

    while True:
        try:
            cmd = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split(maxsplit=2)
        action = parts[0].lower()

        if action == "quit":
            break
        elif action == "add" and len(parts) >= 2:
            name = parts[2] if len(parts) > 2 else ""
            pool.add_backend(parts[1], name)
        elif action == "remove" and len(parts) >= 2:
            pool.remove_backend(parts[1])
            print(f"Removed {parts[1]}")
        elif action == "list":
            s = pool.status()
            print(f"\nBackends: {s['healthy_backends']}/{s['total_backends']} healthy, "
                  f"{s['total_vram_gb']:.1f} GB total VRAM")
            for b in s["backends"]:
                state = "ON " if b["healthy"] else "OFF"
                print(f"  [{state}] {b['name']}: {b['gpu']} ({b['vram_gb']:.1f}GB) "
                      f"— {b['active_jobs']} active, {b['completed']} done")
        elif action == "dub" and len(parts) >= 2:
            result = pool.dub_video(parts[1])
            if result:
                print(f"\nDone! Output: {result}")
        elif action == "parallel" and len(parts) >= 2:
            results = pool.dub_parallel(parts[1])
            if results:
                print(f"\nDone! {len(results)} outputs saved.")
        else:
            print("Unknown command. Try: add, remove, list, dub, parallel, quit")
