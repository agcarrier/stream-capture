# Stream Capture

Record a YouTube stream from its first minute (even while it's still live) and transcribe it locally on an Apple Silicon Mac.

- **Record:** `yt-dlp --live-from-start` grabs the audio from the beginning, then follows the live edge.
- **Transcribe:** `ffmpeg` copies what's captured so far into a clean file, and `mlx_whisper` (Whisper large-v3-turbo) turns it into timestamped text.
- **Keep:** each recording lives in `jobs/<video-id>/` (audio + `.txt`/`.srt`/`.vtt`). That folder is gitignored.

## Setup

```bash
brew install yt-dlp ffmpeg
pip3 install mlx-whisper
mkdir -p ~/.config/stream-capture
echo 'your-password' > ~/.config/stream-capture/password
chmod 600 ~/.config/stream-capture/password
```

Optional phone pings: put an ntfy topic name in `~/.config/stream-capture/ntfy_topic` and subscribe to it in the ntfy app. You get a ping when a transcript finishes or fails.

The app asks for that password and remembers you for 30 days. Changing the file signs every browser out after a restart.

## Run

```bash
python3 app.py
```

Then open http://127.0.0.1:8765. The first transcription downloads the model (about 1.6 GB).
