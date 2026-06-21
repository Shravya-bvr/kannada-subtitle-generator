from flask import Flask, render_template, request, send_file
import os
import subprocess
import shutil
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "input"
OUTPUT_FOLDER = "output"

VIDEO_PATH = os.path.join(UPLOAD_FOLDER, "video.mp4")
SRT_PATH = os.path.join(OUTPUT_FOLDER, "subs_kannada.srt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
print("⏳ Loading Whisper LARGE-V3 (CPU)...")
whisper_model = whisper.load_model("large-v3", device="cpu")
print("✅ Whisper loaded")

print("⏳ Loading NLLB English → Kannada model...")
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
mt_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
print("✅ NLLB loaded")

# ---------------- HELPERS ----------------
def format_time(sec):
    ms = int(sec * 1000)
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms = ms % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def translate_to_kannada(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = mt_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("kan_Knda"),
        max_length=256
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------- ROUTES ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        video.save(VIDEO_PATH)

        print("🎙️ Transcribing & translating to English using Whisper...")
        result = whisper_model.transcribe(
            VIDEO_PATH,
            task="translate"
        )

        segments = result["segments"]

        print("🌐 Translating English → Kannada...")
        with open(SRT_PATH, "w", encoding="utf-8") as f:
            idx = 1
            for seg in segments:
                en_text = seg["text"].strip()
                if not en_text:
                    continue

                kn_text = translate_to_kannada(en_text)

                f.write(f"{idx}\n")
                f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
                f.write(kn_text + "\n\n")
                idx += 1

        print("🎬 Opening VLC with Kannada subtitles...")
        vlc_path = shutil.which("vlc") or r"C:\Program Files\VideoLAN\VLC\vlc.exe"
        if vlc_path and os.path.exists(vlc_path):
            subprocess.Popen([
                vlc_path,
                VIDEO_PATH,
                "--sub-file",
                SRT_PATH
            ])

        return render_template("index.html", done=True)

    return render_template("index.html", done=False)

@app.route("/download")
def download():
    return send_file(SRT_PATH, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)