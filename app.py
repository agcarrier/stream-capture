#!/usr/bin/env python3
"""Stream Capture: record a YouTube stream from its first minute and transcribe it locally.

Run:  python3 app.py   then open http://127.0.0.1:8765
Each recording lives in jobs/<video-id>/ (audio, logs, transcript files).
"""
import hashlib
import hmac
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ROOT / "jobs"
JOBS.mkdir(exist_ok=True)
PORT = int(os.environ.get("PORT", "8765"))
MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

# The password lives outside the repo. Changing it signs every browser out.
PASSWORD_FILE = Path(os.environ.get("PASSWORD_FILE", Path.home() / ".config/stream-capture/password"))
COOKIE = "sc_session"
COOKIE_DAYS = 30


def session_token():
    pw = PASSWORD_FILE.read_text().strip()
    if not pw:
        raise SystemExit("Empty password in %s" % PASSWORD_FILE)
    return hmac.new(pw.encode(), b"stream-capture-session", hashlib.sha256).hexdigest()


LOGIN_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Stream Capture</title>
<style>
:root{--bg:#f4f3ef;--panel:#fff;--ink:#1b1b19;--muted:#6d6b64;--line:#e3e1da;--accent:#e5484d;--accent-ink:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--panel:#1b1b1a;--ink:#ecebe6;--muted:#9a988f;--line:#2c2b29;--accent:#ff5c61;--accent-ink:#160707}}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;padding:16px}
form{width:100%%;max-width:340px;display:grid;gap:12px}h1{font-size:22px;margin:0 0 4px}
input,button{font:inherit;border-radius:10px;padding:12px 14px;border:1px solid var(--line)}
input{background:var(--panel);color:var(--ink)}button{background:var(--accent);color:var(--accent-ink);border-color:transparent;font-weight:500;cursor:pointer}
input:focus-visible,button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}.err{color:var(--accent);font-size:14px;margin:0}
</style></head><body><form method="post" action="/login"><h1>Stream Capture</h1>
<label for="pw" style="color:var(--muted)">Password</label>
<input id="pw" name="password" type="password" autocomplete="current-password" autofocus required>
%s<button>Sign in</button></form></body></html>"""


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

NTFY_FILE = PASSWORD_FILE.parent / "ntfy_topic"  # optional; no file, no phone pings
PUBLIC_URL = os.environ.get("PUBLIC_URL", "https://capture.andrewcarrier.dev")

recorders = {}        # job id -> Popen (this server's own children)
transcribing = set()  # job ids with a transcription in flight
live = {}             # job id -> (offset seconds, segments kept from before) while transcribing
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


def hms(s):
    s = int(s)
    h, rest = divmod(s, 3600)
    return "%d:%02d:%02d" % (h, rest // 60, rest % 60) if h else "%02d:%02d" % (rest // 60, rest % 60)


def stamp(s, sep):
    ms = int(round(s * 1000))
    return "%02d:%02d:%02d%s%03d" % (ms // 3600000, ms // 60000 % 60, ms // 1000 % 60, sep, ms % 1000)


def load_segments(d):
    try:
        data = json.loads((d / "transcript" / "transcript.json").read_text())
    except (FileNotFoundError, ValueError):
        return []
    return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
            for s in data.get("segments", []) if s["text"].strip()]


def write_transcript(d, segs):
    out = d / "transcript"
    out.mkdir(exist_ok=True)
    (out / "transcript.json").write_text(json.dumps({"segments": segs}))
    (out / "transcript.txt").write_text("".join(s["text"] + "\n" for s in segs))
    (out / "transcript.srt").write_text("".join(
        "%d\n%s --> %s\n%s\n\n" % (i, stamp(s["start"], ","), stamp(s["end"], ","), s["text"])
        for i, s in enumerate(segs, 1)))
    (out / "transcript.vtt").write_text("WEBVTT\n\n" + "".join(
        "%s --> %s\n%s\n\n" % (stamp(s["start"], "."), stamp(s["end"], "."), s["text"]) for s in segs))


def notify(title, body, jid=None):
    try:
        topic = NTFY_FILE.read_text().strip()
    except FileNotFoundError:
        return
    headers = {"Title": title}
    if jid:
        headers["Click"] = "%s/#job-%s" % (PUBLIC_URL, jid)
    try:
        urllib.request.urlopen(urllib.request.Request(
            "https://ntfy.sh/" + topic, data=body.encode(), headers=headers), timeout=10)
    except OSError as e:
        print("ntfy failed:", e)


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
        # Only the new audio gets transcribed. The last old line is redone, since it may have
        # been cut off mid-sentence when the previous snapshot was taken.
        old = load_segments(d)
        cut = old[-1]["start"] if old else 0.0
        keep = old[:-1]
        snap = d / "snapshot.wav"
        chunk = d / "chunk.json"
        title = read_meta(d).get("title") or jid
        try:
            live[jid] = (cut, keep)
            # Decode from the cut point into a temp wav; the recorder can keep writing the .part.
            subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", "%.3f" % cut, "-i", str(final or part),
                            "-vn", "-ac", "1", "-ar", "16000", str(snap)], check=True)
            new_len = duration(snap) or 0
            if new_len > 1:
                with open(d / "transcribe.log", "w") as log:
                    code = subprocess.call(
                        [WHISPER, str(snap), "--model", MODEL, "--output-dir", str(d),
                         "--output-name", "chunk", "--output-format", "json"],
                        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                if code != 0:
                    raise RuntimeError((tail(d / "transcribe.log", 1) or ["Transcription failed"])[0])
                new = [{"start": s["start"] + cut, "end": s["end"] + cut, "text": s["text"].strip()}
                       for s in json.loads(chunk.read_text())["segments"] if s["text"].strip()]
                write_transcript(d, keep + new if new else old)
            meta = read_meta(d)
            meta.update(transcribed_seconds=cut + new_len, transcribed_at=time.time(), transcribe_error=None)
            write_meta(d, meta)
            notify("Transcript ready", "%s (%s)" % (title, hms(cut + new_len)), jid)
        except Exception as e:  # surface it in the UI instead of dying silently
            meta = read_meta(d)
            meta["transcribe_error"] = str(e)
            write_meta(d, meta)
            notify("Transcription failed", "%s: %s" % (title, e), jid)
        finally:
            snap.unlink(missing_ok=True)
            chunk.unlink(missing_ok=True)
            live.pop(jid, None)
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
    if d.name in live:  # saved lines before the cut, then Whisper's live output shifted past it
        cut, segs = live[d.name]
        segs = list(segs)
        for line in tail(d / "transcribe.log", 100000):
            m = SEG_RE.match(line.strip())
            if m and m.group(2):
                secs = sum(int(x) * 60 ** i for i, x in enumerate(reversed(m.group(1).split(":"))))
                segs.append({"start": cut + secs, "text": m.group(2)})
    else:
        segs = load_segments(d)
    return {"segments": [{"t": hms(s["start"]), "s": int(s["start"]), "text": s["text"]} for s in segs],
            "transcribing": d.name in transcribing}


def watch_recorders():
    """When a recorder exits (stream over, or stopped), transcribe what it caught."""
    while True:
        time.sleep(15)
        for d in JOBS.iterdir():
            with lock:
                try:
                    meta = read_meta(d)
                except (FileNotFoundError, ValueError):
                    continue
                pid = meta.get("pid")
                if not pid or meta.get("finished_pid") == pid or d.name in transcribing \
                        or recording_alive(d.name, meta):
                    continue
                meta["finished_pid"] = pid
                write_meta(d, meta)
                recorders.pop(d.name, None)
            try:
                start_transcription(d.name)
            except (ValueError, LookupError):
                pass  # nothing was recorded


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

    def signed_in(self):
        cookies = SimpleCookie(self.headers.get("Cookie") or "")
        return COOKIE in cookies and hmac.compare_digest(cookies[COOKIE].value, session_token())

    def login(self, method):
        if method == "GET":
            return self.send(200, (LOGIN_PAGE % "").encode(), "text/html; charset=utf-8")
        length = min(int(self.headers.get("Content-Length") or 0), 4096)
        pw = parse_qs(self.rfile.read(length).decode(errors="replace")).get("password", [""])[0]
        if hmac.compare_digest(pw.encode(), PASSWORD_FILE.read_text().strip().encode()):
            # Secure only through the tunnel (https); Safari drops Secure cookies on http://127.0.0.1.
            secure = "; Secure" if self.headers.get("Cf-Connecting-Ip") else ""
            cookie = "%s=%s; Path=/; Max-Age=%d; HttpOnly; SameSite=Lax%s" % (
                COOKIE, session_token(), COOKIE_DAYS * 86400, secure)
            return self.send(303, b"", "text/plain", {"Location": "/", "Set-Cookie": cookie})
        time.sleep(1)  # ponytail: per-request delay only; add a lockout if the URL ever gets hammered
        self.send(401, (LOGIN_PAGE % '<p class="err">Wrong password.</p>').encode(), "text/html; charset=utf-8")

    def route(self, method):
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if parts == ["login"]:
            return self.login(method)
        if not self.signed_in():
            if method == "GET" and not parts:
                return self.send(303, b"", "text/plain", {"Location": "/login"})
            return self.send(401, {"error": "Sign in first"})
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
    if not PASSWORD_FILE.exists():
        raise SystemExit("Set a password first: put it in %s" % PASSWORD_FILE)
    session_token()
    threading.Thread(target=watch_recorders, daemon=True).start()
    print("Stream Capture on http://127.0.0.1:%d" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
