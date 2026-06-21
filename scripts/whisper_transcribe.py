import whisper
import subprocess
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import timedelta
import torch

# ================= CONFIG =================
VIDEO_PATH = "input/video.mp4"
AUDIO_PATH = "audio/extracted_audio.wav"
OUTPUT_SRT = "output/kannada_subtitles.srt"

WHISPER_MODEL = "large-v3"   # ✅ IMPORTANT
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
# =========================================

# Create folders if not exist
os.makedirs("audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ---------- STEP 1: Extract audio ----------
print("🎧 Extracting audio from video...")

subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-vn",
    "-ac", "1",
    "-ar", "16000",
    "-acodec", "pcm_s16le",
    AUDIO_PATH
], check=True)

print("✅ Audio extracted:", AUDIO_PATH)

# ---------- STEP 2: Load Whisper ----------
print("\n⏳ Loading Whisper LARGE-V3 (CPU)...")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
print("✅ Whisper Large-v3 loaded")

# ---------- STEP 3: Transcribe (Kannada → English) ----------
print("\n🎙️ Transcribing & translating to English...")

result = whisper_model.transcribe(
    AUDIO_PATH,
    task="translate",      # Kannada → English
    language="kn",
    verbose=False
)

segments = result["segments"]
print("✅ Transcription done")

# ---------- STEP 4: Group segments into sentences ----------
def group_segments(segments):
    grouped = []
    current = None

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        if current is None:
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "text": text
            }
        else:
            current["end"] = seg["end"]
            current["text"] += " " + text

        if text.endswith((".", "?", "!")):
            grouped.append(current)
            current = None

    if current:
        grouped.append(current)

    return grouped

sentences = group_segments(segments)
print(f"✅ Grouped into {len(sentences)} subtitle sentences")

# ---------- STEP 5: Load NLLB (English → Kannada) ----------
print("\n⏳ Loading NLLB translation model...")

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device)

tokenizer.src_lang = "eng_Latn"
kan_token_id = tokenizer.convert_tokens_to_ids("kan_Knda")

print("✅ NLLB model ready")

# ---------- STEP 6: Translate ----------
subtitles = []

for idx, s in enumerate(sentences, start=1):
    print(f"🔹 Translating {idx}/{len(sentences)}")

    inputs = tokenizer(
        s["text"],
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=kan_token_id,
        max_length=128
    )

    kannada_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    subtitles.append({
        "index": idx,
        "start": s["start"],
        "end": s["end"],
        "text": kannada_text
    })

print("✅ English → Kannada translation done")

# ---------- STEP 7: Write SRT ----------
def format_time(seconds):
    td = timedelta(seconds=float(seconds))
    total = int(td.total_seconds())
    ms = int((seconds - total) * 1000)

    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60

    return f"{h:02}:{m:02}:{s:02},{ms:03}"

with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
    for sub in subtitles:
        f.write(f"{sub['index']}\n")
        f.write(f"{format_time(sub['start'])} --> {format_time(sub['end'])}\n")
        f.write(sub["text"] + "\n\n")

print("\n🎉 DONE SUCCESSFULLY!")
print("📄 Kannada subtitles saved at:", OUTPUT_SRT)
