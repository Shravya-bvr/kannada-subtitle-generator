# 🎬 Kannada Subtitle Generator

A web app that automatically generates Kannada subtitles for any Indian language video.

## How it works
1. Upload a video (any Indian language)
2. Audio is extracted using ffmpeg
3. DSP-based noise reduction (spectral gating via noisereduce) cleans background noise
4. Whisper large-v3 transcribes and translates cleaned audio to English
5. NLLB-200 translates English → Kannada
6. SRT subtitle file is generated and opened in VLC

## Tech Stack
- Python, Flask
- OpenAI Whisper large-v3
- Facebook NLLB-200 (distilled-600M)
- ffmpeg

## How to run
1. Install dependencies:
pip install -r requirements.txt

2. Run the app:
python app.py

3. Open browser at `http://127.0.0.1:5000`
4. Upload a video and click Generate Subtitles

## Output
- Kannada `.srt` subtitle file
- Auto-opens in VLC with subtitles rendered correctly