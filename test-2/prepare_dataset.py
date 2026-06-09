"""
SARSILMAZ İHA SESLER — Veri Seti Hazırlama
============================================

Ham uzun kayıtları (2-5 dk) → 3 saniyelik örtüşen segmentlere böler.
Veri augmentasyonu uygular.
Sonuçları dataset_sarsilmaz/ klasörüne kaydeder.

Augmentasyon:
  - Pitch shift : ±1, ±2 yarı ton (4 varyant)
  - Speed change: 0.93x, 1.07x (2 varyant)
  - Gain jitter : 0.7x, 0.85x, 1.15x (3 varyant)
  → Her orijinal segmentten 9 augmente varyant üretilir

Segment parametreleri:
  - Uzunluk    : 3 saniye
  - Kayma      : 1 saniye (66% örtüşme)
  - Min enerji : DC+sessizlik filtresi
"""

import os
import sys
import glob
import numpy as np
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "iha_sistemi"))
from config import SAMPLE_RATE

# ─── Parametreler ─────────────────────────────────────────────────────────────
DATASET_SRC   = Path(__file__).parent.parent / "SARSILMAZ İHA SESLER"
DATASET_OUT   = Path(__file__).parent / "dataset_sarsilmaz"
SEG_DURATION  = 3.0               # saniye
HOP_DURATION  = 1.0               # saniye (66% örtüşme)
SEG_LEN       = int(SAMPLE_RATE * SEG_DURATION)
HOP_LEN       = int(SAMPLE_RATE * HOP_DURATION)
ENERGY_THRESH = 0.002             # normalizasyon sonrası RMS eşiği (sessizlik filtresi)

# Augmentasyon parametreleri
# NOT: Pitch-shift drone sınıflandırmasında ZARARLITIR:
#   ÖRÜMCEK BPF'sini ±2 st kaydırınca BÜRKÜT BPF'sine yaklaşıyor → yanlış etiket
# Bu nedenle sadece amplitüd (gain) ve additive noise augmentasyonu kullanılıyor.
PITCH_STEPS   = []                # DEVRE DIŞI — frekans içeriğini değiştirdiği için
SPEED_FACTORS = []                # DEVRE DIŞI — BPF'yi değiştirdiği için
GAIN_FACTORS  = [0.60, 0.75, 0.85, 1.15, 1.30, 1.50]   # Sadece ses şiddeti
NOISE_SNRS    = [25, 15, 8]       # dB — AWGN gürültüsü ekleme (güvenli augment)


# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _load(fpath: str) -> np.ndarray:
    """Ses yükle, mono'ya çevir, 16kHz yeniden örnekle."""
    audio, sr = sf.read(fpath, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float64)

    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE
            ).astype(np.float64)
        except ImportError:
            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)), audio
            )
    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak
    return audio


def _pitch_shift(audio: np.ndarray, n_steps: int) -> np.ndarray:
    """Pitch shift: n_steps yarı ton yukarı/aşağı."""
    try:
        import librosa
        return librosa.effects.pitch_shift(
            audio.astype(np.float32), sr=SAMPLE_RATE, n_steps=n_steps
        ).astype(np.float64)
    except ImportError:
        # Librosa yoksa frekans kaymasını yaklaşık simüle et
        factor = 2 ** (n_steps / 12.0)
        new_len = int(len(audio) / factor)
        resampled = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)), audio
        )
        if len(resampled) >= len(audio):
            return resampled[:len(audio)]
        else:
            return np.pad(resampled, (0, len(audio) - len(resampled)))


def _speed_change(audio: np.ndarray, factor: float) -> np.ndarray:
    """Zaman uzatma/kısaltma (pitch değişmeden)."""
    try:
        import librosa
        return librosa.effects.time_stretch(
            audio.astype(np.float32), rate=factor
        ).astype(np.float64)
    except ImportError:
        new_len = int(len(audio) / factor)
        resampled = np.interp(
            np.linspace(0, len(audio) - 1, new_len),
            np.arange(len(audio)), audio
        )
        if len(resampled) >= len(audio):
            return resampled[:len(audio)]
        else:
            return np.pad(resampled, (0, len(audio) - len(resampled)))


def _rms(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg ** 2)))


def _save_segment(seg: np.ndarray, out_path: str):
    """Segmenti normalize edip WAV olarak kaydet."""
    peak = np.max(np.abs(seg))
    if peak > 0:
        seg = seg / peak * 0.9
    sf.write(out_path, seg.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")


# ─── Ana Segmentasyon Fonksiyonu ──────────────────────────────────────────────

def segment_file(audio: np.ndarray, out_dir: str, base_name: str,
                 augment: bool = True):
    """
    Uzun sesi örtüşen 3-saniyelik segmentlere böl.
    Sessizlik filtresi uygula.
    Augmentasyon varsa ek varyantlar üret.
    """
    os.makedirs(out_dir, exist_ok=True)
    saved = 0

    # Örtüşen segment indeksleri
    starts = list(range(0, len(audio) - SEG_LEN + 1, HOP_LEN))

    for si, start in enumerate(starts):
        seg = audio[start:start + SEG_LEN].copy()

        # Sessizlik filtresi
        if _rms(seg) < ENERGY_THRESH:
            continue

        # Orijinal segment kaydet
        fname = f"{base_name}_seg{si:04d}.wav"
        _save_segment(seg, os.path.join(out_dir, fname))
        saved += 1

        if not augment:
            continue

        # Gain augmentation (frekans içeriğini değiştirmiyor → güvenli)
        for gi, gain in enumerate(GAIN_FACTORS):
            aug = seg * gain
            if _rms(aug) >= ENERGY_THRESH:
                _save_segment(aug, os.path.join(out_dir,
                    f"{base_name}_seg{si:04d}_g{gi}.wav"))
                saved += 1

        # AWGN gürültüsü ekleme (gerçek dünya simülasyonu, frekans değiştirmiyor)
        rng = np.random.default_rng(si * 997 + 42)
        for ni, snr_db in enumerate(NOISE_SNRS):
            sig_power   = np.mean(seg ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise       = rng.normal(0, np.sqrt(max(noise_power, 1e-12)), len(seg))
            aug         = seg + noise
            if _rms(aug) >= ENERGY_THRESH:
                _save_segment(aug, os.path.join(out_dir,
                    f"{base_name}_seg{si:04d}_n{ni}.wav"))
                saved += 1

    return saved


# ─── Ana Akış ─────────────────────────────────────────────────────────────────

def prepare(augment_minority: bool = True):
    """
    Tüm sınıflar için dataset_sarsilmaz/ hazırla.

    augment_minority:
        True  → Az veri olan sınıflar (BÜRKÜT, HEXA, ÖRÜMCEK) augmente edilir
        False → Sadece orijinal segmentler
    """
    print(f"\n{'═'*62}")
    print(f"  VERİ SETİ HAZIRLAMA — 3s Segment + Augmentasyon")
    print(f"{'═'*62}")
    print(f"  Kaynak  : {DATASET_SRC}")
    print(f"  Çıktı   : {DATASET_OUT}")
    print(f"  Segment : {SEG_DURATION}s  |  Kayma: {HOP_DURATION}s  |  SR: {SAMPLE_RATE}Hz")

    class_dirs = sorted([
        d for d in DATASET_SRC.iterdir()
        if d.is_dir() and d.name != "GÖRSELLER"
    ])

    # Sınıf başına orijinal süreyi hesapla (hangileri minority?)
    class_durations = {}
    for d in class_dirs:
        files = glob.glob(str(d / "*.wav")) + glob.glob(str(d / "*.WAV"))
        total_s = 0
        for f in files:
            info = sf.info(f)
            total_s += info.duration
        class_durations[d.name] = total_s

    max_dur = max(class_durations.values())
    print(f"\n  Süre özeti:")
    for cn, dur in class_durations.items():
        is_minority = dur < max_dur * 0.6
        print(f"    {cn:<14}: {dur/60:.1f} dk  {'← augmente edilecek' if is_minority else ''}")

    total_saved = {}

    for class_dir in class_dirs:
        cname   = class_dir.name
        out_dir = str(DATASET_OUT / cname)
        files   = sorted(
            glob.glob(str(class_dir / "*.wav")) +
            glob.glob(str(class_dir / "*.WAV")) +
            glob.glob(str(class_dir / "*.flac"))
        )

        is_minority = class_durations[cname] < max_dur * 0.6
        do_augment  = augment_minority and is_minority

        print(f"\n  [{cname}]  {len(files)} dosya  {'(augmentasyon açık)' if do_augment else '(sadece segmentasyon)'}")
        saved = 0

        for fpath in files:
            fname = Path(fpath).stem
            print(f"    Yükleniyor: {Path(fpath).name} ...", end=" ", flush=True)
            try:
                audio = _load(fpath)
                n = segment_file(audio, out_dir, fname, augment=do_augment)
                print(f"{n} segment")
                saved += n
            except Exception as e:
                print(f"HATA: {e}")

        total_saved[cname] = saved
        print(f"    → Toplam {cname}: {saved} segment")

    print(f"\n{'═'*62}")
    print(f"  ÖZET")
    print(f"{'═'*62}")
    for cn, n in total_saved.items():
        print(f"  {cn:<14}: {n:>5} segment")
    print(f"  {'TOPLAM':<14}: {sum(total_saved.values()):>5} segment")
    print(f"\n  Çıktı klasörü: {DATASET_OUT}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-aug", action="store_true", help="Augmentasyonu devre dışı bırak")
    args = p.parse_args()
    prepare(augment_minority=not args.no_aug)
