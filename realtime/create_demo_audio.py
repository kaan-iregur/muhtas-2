"""
Demo Ses Dosyası Oluşturucu
============================
Sarsilmaz ham ses dosyalarından blok yapılı demo karması üretir.

Yapı:
  [30s BÜRKÜT] [2s sessiz] [30s DİNOZOR] [2s sessiz]
  [30s HEXA]   [2s sessiz] [30s ÖRÜMCEK]

Çıktı:
  demo_audio.wav          — 16kHz mono WAV
  demo_ground_truth.json  — her saniyede hangi sınıf

Kullanım:
  python create_demo_audio.py
"""

import os
import sys
import json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SAMPLE_RATE, DEMO_AUDIO_PATH, DEMO_GT_PATH,
    DEMO_BLOCK_DURATION, DEMO_SILENCE_DURATION, SARSILMAZ_FILES,
)

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def resample_to_mono(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Stereo→mono + yeniden örnekleme."""
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if orig_sr == target_sr:
        return audio
    if _HAS_LIBROSA:
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    # librosa yoksa doğrusal interpolasyon
    new_len = int(len(audio) * target_sr / orig_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, new_len),
        np.arange(len(audio)),
        audio.astype(np.float64),
    ).astype(np.float32)


def extract_middle_block(file_path: str, duration: float) -> np.ndarray:
    """
    Ses dosyasının ortasından 'duration' saniyelik blok alır.
    Dosya yetmezse tüm dosyayı döndürür (tekrarlanır).
    """
    audio, sr = sf.read(file_path, always_2d=True, dtype="float32")
    audio = resample_to_mono(audio, sr, SAMPLE_RATE)

    block_len = int(duration * SAMPLE_RATE)
    total_len = len(audio)

    if total_len <= block_len:
        # Dosya kısa — döngüsel tekrar
        repeats = (block_len // total_len) + 2
        audio = np.tile(audio, repeats)

    # Ortadan al
    start = (len(audio) - block_len) // 2
    block = audio[start: start + block_len].copy()

    # Genlik normalize
    peak = np.max(np.abs(block))
    if peak > 1e-8:
        block /= peak
    block *= 0.85  # biraz boşluk bırak

    return block.astype(np.float32)


def create_demo():
    out_dir = os.path.dirname(DEMO_AUDIO_PATH)
    os.makedirs(out_dir, exist_ok=True)

    block_len   = int(DEMO_BLOCK_DURATION   * SAMPLE_RATE)
    silence_len = int(DEMO_SILENCE_DURATION * SAMPLE_RATE)
    silence     = np.zeros(silence_len, dtype=np.float32)

    classes_ordered = ["BÜRKÜT", "DİNOZOR", "HEXA", "ÖRÜMCEK"]
    segments        = []
    ground_truth    = []   # [{start_s, end_s, class}]

    print("Demo ses dosyası oluşturuluyor...\n")
    cursor = 0.0  # saniye cinsinden konum

    for i, cls in enumerate(classes_ordered):
        fpath = SARSILMAZ_FILES[cls]
        if not os.path.exists(fpath):
            print(f"  [UYARI] {cls} ses dosyası bulunamadı: {fpath}")
            continue

        print(f"  [{cls}] {os.path.basename(fpath)} okunuyor...")
        block = extract_middle_block(fpath, DEMO_BLOCK_DURATION)
        segments.append(block)

        ground_truth.append({
            "class":   cls,
            "start_s": round(cursor, 3),
            "end_s":   round(cursor + DEMO_BLOCK_DURATION, 3),
        })
        print(f"    → {cursor:.1f}s – {cursor + DEMO_BLOCK_DURATION:.1f}s")
        cursor += DEMO_BLOCK_DURATION

        if i < len(classes_ordered) - 1:
            segments.append(silence)
            cursor += DEMO_SILENCE_DURATION

    if not segments:
        print("Hiç ses dosyası bulunamadı, çıkılıyor.")
        return

    final_audio = np.concatenate(segments)
    sf.write(DEMO_AUDIO_PATH, final_audio, SAMPLE_RATE, subtype="PCM_16")

    with open(DEMO_GT_PATH, "w", encoding="utf-8") as f:
        json.dump({"sample_rate": SAMPLE_RATE, "blocks": ground_truth}, f,
                  ensure_ascii=False, indent=2)

    total = len(final_audio) / SAMPLE_RATE
    print(f"\nToplam süre : {total:.1f} saniye")
    print(f"Ses dosyası : {DEMO_AUDIO_PATH}")
    print(f"Ground truth: {DEMO_GT_PATH}")
    print("\nTamamlandı.")


if __name__ == "__main__":
    create_demo()
