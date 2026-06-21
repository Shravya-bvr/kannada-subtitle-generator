import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
import os
import sys

# ---------------- CONFIG ----------------
INPUT_AUDIO = "audio/extracted_audio.wav"
OUTPUT_AUDIO = "audio/clean_audio.wav"
TARGET_SR = 16000
# ---------------------------------------

def main():
    # Check input audio
    if not os.path.exists(INPUT_AUDIO):
        print("❌ ERROR: audio/extracted_audio.wav not found")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs("audio", exist_ok=True)

    print("🔊 Loading audio...")
    audio, sr = librosa.load(INPUT_AUDIO, sr=TARGET_SR, mono=True)

    duration = len(audio) / sr
    print(f"✅ Audio loaded ({duration:.2f}s, {sr}Hz)")

    # ---- SAFE NOISE SAMPLE (FIRST 0.3s ONLY) ----
    noise_sample = audio[: int(0.3 * sr)]

    print("🧹 Applying SAFE noise reduction...")
    reduced_audio = nr.reduce_noise(
        y=audio,
        y_noise=noise_sample,
        sr=sr,
        stationary=True,
        prop_decrease=0.6   # 🔥 Balanced (does NOT kill silence)
    )

    # ---- VERY LIGHT NORMALIZATION ----
    peak = np.max(np.abs(reduced_audio))
    if peak > 0:
        reduced_audio = (reduced_audio / peak) * 0.9

    print("💾 Saving clean audio...")
    sf.write(
        OUTPUT_AUDIO,
        reduced_audio,
        sr,
        subtype="PCM_16"
    )

    print("✅ Noise reduction completed successfully!")
    print(f"📁 Clean audio saved at: {OUTPUT_AUDIO}")

if __name__ == "__main__":
    main()
