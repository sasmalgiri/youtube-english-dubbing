"""GPU Pool Manager — distribute dubbing work across multiple free GPU backends.

Supports:
- Local GPU (if NVIDIA card installed)
- Multiple Colab/Kaggle backends (user starts manually, registers URL)
- Kaggle API (auto-submit GPU kernels)
- Lightning AI (auto-spin up studios)

Usage:
    pool = GPUPool()
    pool.add_backend("http://colab-ngrok-url.ngrok.io")  # Colab backend
    pool.add_backend("http://kaggle-ngrok-url.ngrok.io")  # Kaggle backend
    pool.add_backend("http://localhost:8000")              # Local backend

    # Distribute split parts across all backends
    results = pool.dub_parallel(video_url, parts=20, target_language="en")
"""

import os
import time
import json
import threading
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    requests = None


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
        """Ping the backend to check if it's alive and has GPU."""
        try:
            resp = requests.get(f"{self.url}/api/health", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                self.is_healthy = True
                self.gpu_type = data.get("gpu", "unknown")
                self.vram_gb = data.get("vram_gb", 0)
                self.name = self.name or data.get("hostname", self.url)
                self.last_health_check = time.time()
                return True
        except Exception:
            pass
        self.is_healthy = False
        return False


@dataclass
class PoolJob:
    """Tracks a part being processed on a backend."""
    part_num: int
    backend: GPUBackend
    job_id: str = ""
    state: str = "pending"  # pending, running, done, failed
    progress: float = 0.0
    result_url: str = ""
    error: str = ""


class GPUPool:
    """Manages multiple GPU backends and distributes work across them."""

    def __init__(self):
        self.backends: List[GPUBackend] = []
        self._lock = threading.Lock()
        self._on_progress = None

    def add_backend(self, url: str, name: str = "") -> GPUBackend:
        """Register a GPU backend (Colab, Kaggle, local, etc.)."""
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
        """Remove a backend from the pool."""
        url = url.rstrip("/")
        with self._lock:
            self.backends = [b for b in self.backends if b.url != url]

    def get_healthy_backends(self) -> List[GPUBackend]:
        """Return all healthy backends, refreshing stale health checks."""
        now = time.time()
        for b in self.backends:
            if now - b.last_health_check > 30:  # Re-check every 30s
                b.health_check()
        return [b for b in self.backends if b.is_healthy]

    def _pick_backend(self) -> Optional[GPUBackend]:
        """Pick the least-loaded healthy backend."""
        healthy = self.get_healthy_backends()
        if not healthy:
            return None
        # Sort by active jobs (least busy first), then by VRAM (more is better)
        healthy.sort(key=lambda b: (b.active_jobs, -b.vram_gb))
        return healthy[0]

    def _submit_part(self, backend: GPUBackend, video_url: str, part_num: int,
                     total_parts: int, split_start_sec: float, split_duration_sec: float,
                     settings: Dict) -> PoolJob:
        """Submit a single part to a backend for dubbing."""
        job = PoolJob(part_num=part_num, backend=backend)

        try:
            # Submit dubbing job to the backend
            payload = {
                "url": video_url,
                "source_language": settings.get("source_language", "auto"),
                "target_language": settings.get("target_language", "en"),
                "asr_model": settings.get("asr_model", "large-v3-turbo"),
                "translation_engine": settings.get("translation_engine", "auto"),
                "split_start": split_start_sec,
                "split_end": split_start_sec + split_duration_sec,
                **{k: v for k, v in settings.items()
                   if k not in ("source_language", "target_language", "asr_model",
                                "translation_engine", "split_start", "split_end")},
            }

            resp = requests.post(f"{backend.url}/api/jobs", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            job.job_id = data.get("job_id", "")
            job.state = "running"
            backend.active_jobs += 1

            print(f"[Pool] Part {part_num}/{total_parts} → {backend.name} (job {job.job_id})")

        except Exception as e:
            job.state = "failed"
            job.error = str(e)
            print(f"[Pool] Part {part_num} FAILED to submit to {backend.name}: {e}")

        return job

    def _wait_for_job(self, job: PoolJob, timeout: int = 7200) -> PoolJob:
        """Poll a backend until the job completes or times out."""
        if job.state == "failed":
            return job

        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = requests.get(
                    f"{job.backend.url}/api/jobs/{job.job_id}/status",
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    job.state = data.get("state", "running")
                    job.progress = data.get("overall_progress", 0)

                    if job.state == "done":
                        job.result_url = f"{job.backend.url}/api/jobs/{job.job_id}/download"
                        job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
                        job.backend.total_completed += 1
                        print(f"[Pool] Part {job.part_num} DONE on {job.backend.name}")
                        return job

                    if job.state == "error":
                        job.error = data.get("message", "Unknown error")
                        job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
                        print(f"[Pool] Part {job.part_num} ERROR on {job.backend.name}: {job.error}")
                        return job

            except Exception:
                # Backend might be temporarily unreachable
                if not job.backend.health_check():
                    job.state = "failed"
                    job.error = "Backend went offline"
                    job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
                    return job

            time.sleep(10)  # Poll every 10 seconds

        job.state = "failed"
        job.error = "Timeout"
        job.backend.active_jobs = max(0, job.backend.active_jobs - 1)
        return job

    def _download_result(self, job: PoolJob, output_dir: Path) -> Optional[Path]:
        """Download the dubbed video from a backend."""
        if not job.result_url:
            return None
        try:
            resp = requests.get(job.result_url, stream=True, timeout=300)
            resp.raise_for_status()
            out_path = output_dir / f"part_{job.part_num:02d}.mp4"
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[Pool] Downloaded part {job.part_num} → {out_path}")
            return out_path
        except Exception as e:
            print(f"[Pool] Download failed for part {job.part_num}: {e}")
            return None

    def dub_parallel(self, video_url: str, split_minutes: int = 30,
                     video_duration_sec: float = 0,
                     settings: Optional[Dict] = None,
                     output_dir: str = "gpu_pool_output",
                     max_retries: int = 2) -> List[Path]:
        """Dub a video by splitting and distributing parts across all backends.

        Args:
            video_url: YouTube URL or direct video URL
            split_minutes: Split every N minutes
            video_duration_sec: Total video duration (0 = auto-detect)
            settings: Dubbing settings dict
            output_dir: Where to save results
            max_retries: Retry failed parts on other backends

        Returns:
            List of output file paths (ordered by part number)
        """
        settings = settings or {}
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Check backends
        healthy = self.get_healthy_backends()
        if not healthy:
            raise RuntimeError("No healthy GPU backends available. Add backends first.")

        print(f"\n{'='*60}")
        print(f"GPU POOL: {len(healthy)} backends available")
        for b in healthy:
            print(f"  - {b.name}: {b.gpu_type} ({b.vram_gb:.1f}GB VRAM)")
        print(f"{'='*60}\n")

        # Calculate parts
        if video_duration_sec <= 0:
            # Try to get duration from first backend
            try:
                resp = requests.post(
                    f"{healthy[0].url}/api/video-info",
                    json={"url": video_url}, timeout=30)
                video_duration_sec = resp.json().get("duration", 3600)
            except Exception:
                video_duration_sec = 3600  # Assume 1hr if can't detect
                print(f"[Pool] Can't detect duration, assuming {video_duration_sec}s")

        split_sec = split_minutes * 60
        num_parts = max(1, int(video_duration_sec / split_sec) + (1 if video_duration_sec % split_sec else 0))

        print(f"[Pool] Video: {video_duration_sec/60:.0f} min → {num_parts} parts of {split_minutes} min")
        print(f"[Pool] Distributing across {len(healthy)} backends...\n")

        # Submit all parts in parallel across backends
        results: Dict[int, PoolJob] = {}
        with ThreadPoolExecutor(max_workers=len(healthy) * 2) as pool:
            futures = {}
            for part_idx in range(num_parts):
                backend = self._pick_backend()
                if not backend:
                    print(f"[Pool] No backend available for part {part_idx+1}, queuing...")
                    time.sleep(5)
                    backend = self._pick_backend()
                    if not backend:
                        continue

                start_sec = part_idx * split_sec
                dur_sec = min(split_sec, video_duration_sec - start_sec)

                fut = pool.submit(
                    self._submit_and_wait,
                    backend, video_url, part_idx + 1, num_parts,
                    start_sec, dur_sec, settings
                )
                futures[fut] = part_idx + 1

            # Collect results
            for fut in as_completed(futures):
                part_num = futures[fut]
                job = fut.result()
                results[part_num] = job

        # Retry failed parts on different backends
        for retry in range(max_retries):
            failed = [pn for pn, j in results.items() if j.state == "failed"]
            if not failed:
                break
            print(f"\n[Pool] Retry {retry+1}: {len(failed)} failed parts...")
            for part_num in failed:
                backend = self._pick_backend()
                if not backend:
                    continue
                start_sec = (part_num - 1) * split_sec
                dur_sec = min(split_sec, video_duration_sec - start_sec)
                job = self._submit_and_wait(
                    backend, video_url, part_num, num_parts,
                    start_sec, dur_sec, settings
                )
                results[part_num] = job

        # Download all completed parts
        print(f"\n[Pool] Downloading results...")
        output_files = []
        for part_num in sorted(results.keys()):
            job = results[part_num]
            if job.state == "done":
                path = self._download_result(job, out_path)
                if path:
                    output_files.append(path)
            else:
                print(f"[Pool] Part {part_num}: {job.state} — {job.error}")

        # Summary
        done = sum(1 for j in results.values() if j.state == "done")
        failed = sum(1 for j in results.values() if j.state == "failed")
        print(f"\n{'='*60}")
        print(f"GPU POOL COMPLETE: {done}/{num_parts} parts dubbed")
        if failed:
            print(f"  FAILED: {failed} parts")
        print(f"  Output: {out_path}")
        print(f"{'='*60}")

        return output_files

    def _submit_and_wait(self, backend, video_url, part_num, total_parts,
                         start_sec, dur_sec, settings):
        """Submit a part and wait for completion."""
        job = self._submit_part(
            backend, video_url, part_num, total_parts,
            start_sec, dur_sec, settings
        )
        if job.state == "running":
            job = self._wait_for_job(job)
        return job

    def status(self) -> Dict:
        """Get pool status summary."""
        healthy = self.get_healthy_backends()
        return {
            "total_backends": len(self.backends),
            "healthy_backends": len(healthy),
            "total_vram_gb": sum(b.vram_gb for b in healthy),
            "active_jobs": sum(b.active_jobs for b in healthy),
            "total_completed": sum(b.total_completed for b in self.backends),
            "backends": [
                {
                    "name": b.name,
                    "url": b.url,
                    "gpu": b.gpu_type,
                    "vram_gb": b.vram_gb,
                    "healthy": b.is_healthy,
                    "active_jobs": b.active_jobs,
                    "completed": b.total_completed,
                }
                for b in self.backends
            ],
        }


# ── CLI usage ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("GPU Pool Manager — Zero-Spend English Dubbing")
    print("=" * 50)

    pool = GPUPool()

    # Auto-detect local backend
    try:
        resp = requests.get("http://localhost:8000/api/health", timeout=3)
        if resp.status_code == 200:
            pool.add_backend("http://localhost:8000", "Local")
    except Exception:
        pass

    # Interactive mode
    print("\nCommands:")
    print("  add <url> [name]  — Add a GPU backend")
    print("  list              — Show all backends")
    print("  dub <youtube-url> — Start parallel dubbing")
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
        elif action == "list":
            status = pool.status()
            print(f"\nBackends: {status['healthy_backends']}/{status['total_backends']} healthy")
            print(f"Total VRAM: {status['total_vram_gb']:.1f} GB")
            for b in status["backends"]:
                state = "ONLINE" if b["healthy"] else "OFFLINE"
                print(f"  [{state}] {b['name']}: {b['gpu']} ({b['vram_gb']:.1f}GB) "
                      f"— {b['active_jobs']} active, {b['completed']} done")
        elif action == "dub" and len(parts) >= 2:
            url = parts[1]
            try:
                results = pool.dub_parallel(url, split_minutes=30)
                print(f"\nDone! {len(results)} parts saved.")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Unknown command. Try: add, list, dub, quit")
