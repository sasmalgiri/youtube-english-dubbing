import re
from pathlib import Path
from typing import List, Dict


def _parse_time(ts: str) -> float:
    """Parse SRT timestamp '00:01:23,456' to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def parse_srt(srt_path: Path, text_key: str = "text_translated") -> List[Dict]:
    """Parse an SRT file into segments.
    Extracts [SPEAKER_XX] labels if present in the text.
    Returns: [{"start": float, "end": float, text_key: str, "speaker_id": str (optional)}, ...]
    """
    content = srt_path.read_text(encoding="utf-8-sig")
    segments = []
    # Split on blank lines to get blocks
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        # Line 1: index (skip), Line 2: timestamps, Line 3+: text
        ts_match = re.search(r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})", lines[1])
        if not ts_match:
            continue
        start = _parse_time(ts_match.group(1))
        end = _parse_time(ts_match.group(2))
        text = " ".join(lines[2:]).strip()
        if text:
            seg = {"start": start, "end": end, text_key: text}
            # Extract [SPEAKER_XX] label if present
            speaker_match = re.match(r"\[(SPEAKER_\d+)\]\s*(.+)", text)
            if speaker_match:
                seg["speaker_id"] = speaker_match.group(1)
                clean_text = speaker_match.group(2).strip()
                seg[text_key] = clean_text
            # Also set "text" key so downstream code that accesses seg["text"] works
            if text_key != "text":
                seg["text"] = seg[text_key]
            segments.append(seg)
    return segments


def _fmt_time(t: float) -> str:
    if t < 0:
        t = 0
    ms = int(round(t * 1000.0))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(segments: List[Dict], out_path: Path, text_key: str = "text",
              include_speaker: bool = False):
    lines = []
    idx = 0
    for seg in sorted(segments, key=lambda s: s["start"]):
        start = _fmt_time(seg["start"])
        end = _fmt_time(seg["end"])
        text = seg.get(text_key, "").strip()
        if text:
            idx += 1
            if include_speaker and "speaker_id" in seg:
                text = f"[{seg['speaker_id']}] {text}"
            lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
