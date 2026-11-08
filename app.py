#!/usr/bin/env python3
"""Stream Capture: record a YouTube stream from its first minute and transcribe it locally.

Run:  python3 app.py   then open http://127.0.0.1:8765
Each recording lives in jobs/<video-id>/ (audio, logs, transcript files).
"""
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"
JOBS.mkdir(exist_ok=True)
PORT = int(os.environ.get("PORT", "8765"))
MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")


def tool(name, *fallbacks):
    found = shutil.which(name)
    if found:
        return found
    for f in fallbacks:
        f = os.path.expanduser(f)
        if os.path.exists(f):
            return f
    return name


YTDLP = tool("yt-dlp", "/opt/homebrew/bin/yt-dlp")
FFMPEG = tool("ffmpeg", "/opt/homebrew/bin/ffmpeg")
FFPROBE = tool("ffprobe", "/opt/homebrew/bin/ffprobe")
WHISPER = tool("mlx_whisper", "~/Library/Python/3.9/bin/mlx_whisper")

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
SEG_RE = re.compile(r"^\[((?:\d+:)?\d+:\d+)\.\d+ --> [^\]]+\]\s*(.*)$")

recorders = {}        # job id -> Popen (this server's own children)
transcribing = set()  # job ids with a transcription in flight
_dur_cache = {}
lock = threading.Lock()


# ---------- job files ----------

def job_dir(jid):
    if not ID_RE.match(jid or "") or not (JOBS / jid / "meta.json").exists():
        raise LookupError("No such recording")
    return JOBS / jid


def read_meta(d):
    return json.loads((d / "meta.json").read_text())


def write_meta(d, meta):
    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def audio_files(d):
    """(finished file or None, in-progress .part or None)."""
    final = part = None
    for f in d.glob("audio.*"):
        if "-Frag" in f.name or f.suffix == ".ytdl":
            continue
        if f.suffix == ".part":
            part = f
        else:
            final = f
    return final, part


def duration(f):
    if not f or not f.exists():
        return None
    key = (str(f), f.stat().st_size)
    if key not in _dur_cache:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(f)], capture_output=True, text=True)
        try:
            _dur_cache[key] = float(out.stdout.strip())
        except ValueError:
            _dur_cache[key] = None
    return _dur_cache[key]


def tail(path, n=40):
    try:
        lines = path.read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return []
    return lines[-n:]


def recording_alive(jid, meta):
    p = recorders.get(jid)
    if p is not None:
        return p.poll() is None
    pid = meta.get("pid")  # a recorder left running by an earlier server run
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------- actions ----------

def start_recording(url):
    if not re.match(r"^https?://", url or ""):
        raise ValueError("Paste a full video URL (https://...)")
    probe = subprocess.run(
        [YTDLP, "--no-warnings", "--no-playlist", "--skip-download",
         "--print", "%(id)s\t%(title)s\t%(live_status)s", "--", url],
        capture_output=True, text=True, timeout=90)
    if probe.returncode != 0 or not probe.stdout.strip():
        err = (probe.stderr.strip().splitlines() or ["Couldn't read that URL"])[-1]
        raise ValueError(err.replace("ERROR: ", ""))
    vid, title, live = (probe.stdout.strip().splitlines()[0].split("\t") + ["", ""])[:3]
    if not ID_RE.match(vid):
        raise ValueError("Unexpected video id")
    d = JOBS / vid
    d.mkdir(exist_ok=True)
    meta = read_meta(d) if (d / "meta.json").exists() else {"started": time.time()}
    with lock:
        if recording_alive(vid, meta):
            raise ValueError("Already recording this one")
        meta.update(id=vid, url=url, title=title, live_status=live)
        log = open(d / "record.log", "a")
        p = subprocess.Popen(
            [YTDLP, "--live-from-start", "--no-playlist", "--newline", "-f", "bestaudio",
             "-o", "audio.%(ext)s", "--", url],
            cwd=d, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        recorders[vid] = p
        meta["pid"] = p.pid
        write_meta(d, meta)
    return vid


def stop_recording(jid):
    d = job_dir(jid)
    meta = read_meta(d)
    if not recording_alive(jid, meta):
        return
    p = recorders.get(jid)
    if p:
        p.send_signal(signal.SIGINT)
    else:
        os.kill(meta["pid"], signal.SIGINT)


def start_transcription(jid):
    d = job_dir(jid)
    final, part = audio_files(d)
    if not (final or part):
        raise ValueError("Nothing recorded yet")
    with lock:
        if jid in transcribing:
            raise ValueError("Already transcribing")
        transcribing.add(jid)

    def run():
        src = final
        try:
            if not src:
                # Copy what's on disk so far into a playable file; the recorder keeps writing.
                src = d / ("snapshot" + Path(part.stem).suffix)
                subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(part), "-vn", "-c", "copy",
                                str(src)], check=True)
            covered = duration(src)
            with open(d / "transcribe.log", "w") as log:
                code = subprocess.call(
                    [WHISPER, str(src), "--model", MODEL, "--output-dir", str(d / "transcript"),
                     "--output-name", "transcript", "--output-format", "all"],
                    stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
            meta = read_meta(d)
            if code == 0:
                meta.update(transcribed_seconds=covered, transcribed_at=time.time(),
                            transcribe_error=None)
            else:
                meta["transcribe_error"] = (tail(d / "transcribe.log", 1) or ["Transcription failed"])[0]
            write_meta(d, meta)
        except Exception as e:  # surface it in the UI instead of dying silently
            meta = read_meta(d)
            meta["transcribe_error"] = str(e)
            write_meta(d, meta)
        finally:
            if src is not final and src is not None:
                src.unlink(missing_ok=True)  # the snapshot is a throwaway copy of the .part
            transcribing.discard(jid)

    threading.Thread(target=run, daemon=True).start()


# ---------- views ----------

def job_info(d):
    meta = read_meta(d)
    jid = d.name
    final, part = audio_files(d)
    rec = recording_alive(jid, meta)
    log = tail(d / "record.log", 30)
    progress = next((l for l in reversed(log) if l.startswith("[download]")), "")
    error = None
    if not rec and not final:
        error = next((l.replace("ERROR: ", "") for l in reversed(log) if l.startswith("ERROR")), None)
    state = "recording" if rec else "complete" if final else "stopped" if part else "empty"
    return {
        "id": jid,
        "title": meta.get("title") or jid,
        "url": meta.get("url"),
        "live_status": meta.get("live_status"),
        "started": meta.get("started"),
        "state": state,
        "captured_seconds": duration(final or part),
        "progress": progress.replace("[download]", "").strip(),
        "error": error,
        "transcribing": jid in transcribing,
        "transcribed_seconds": meta.get("transcribed_seconds"),
        "transcribed_at": meta.get("transcribed_at"),
        "transcribe_error": meta.get("transcribe_error"),
        "has_transcript": (d / "transcript" / "transcript.txt").exists(),
    }


def transcript(d):
    segs = []
    for line in tail(d / "transcribe.log", 100000):
        m = SEG_RE.match(line.strip())
        if m and m.group(2):
            segs.append({"t": m.group(1), "text": m.group(2)})
    return {"segments": segs, "transcribing": d.name in transcribing}


# ---------- http ----------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json", extra=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def route(self, method):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        try:
            if method == "GET" and not parts:
                return self.send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            if parts[:2] != ["api", "jobs"]:
                return self.send(404, {"error": "Not found"})
            rest = parts[2:]
            if method == "GET" and not rest:
                dirs = [d for d in JOBS.iterdir() if (d / "meta.json").exists()]
                jobs = sorted((job_info(d) for d in dirs), key=lambda j: -(j["started"] or 0))
                return self.send(200, {"jobs": jobs})
            if method == "POST" and not rest:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                return self.send(200, {"id": start_recording(body.get("url", "").strip())})
            d = job_dir(rest[0])
            action = rest[1] if len(rest) > 1 else ""
            if method == "GET" and action == "transcript":
                return self.send(200, transcript(d))
            if method == "GET" and action == "download" and len(rest) == 3 and rest[2] in ("txt", "srt", "vtt"):
                f = d / "transcript" / ("transcript." + rest[2])
                if not f.exists():
                    return self.send(404, {"error": "No transcript yet"})
                return self.send(200, f.read_bytes(), "text/plain; charset=utf-8",
                                 {"Content-Disposition": 'attachment; filename="%s.%s"' % (d.name, rest[2])})
            if method == "POST" and action == "stop":
                stop_recording(d.name)
                return self.send(200, {"ok": True})
            if method == "POST" and action == "resume":
                return self.send(200, {"id": start_recording(read_meta(d)["url"])})
            if method == "POST" and action == "transcribe":
                start_transcription(d.name)
                return self.send(200, {"ok": True})
            return self.send(404, {"error": "Not found"})
        except LookupError as e:
            self.send(404, {"error": str(e)})
        except (ValueError, subprocess.TimeoutExpired) as e:
            self.send(400, {"error": str(e)})

    def do_GET(self):
        self.route("GET")

    def do_POST(self):
        self.route("POST")


if __name__ == "__main__":
    print("Stream Capture on http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
