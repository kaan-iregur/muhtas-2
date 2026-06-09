"""
IHA Ses Sınıflandırma Sistemi - Özellik Çıkarma Modülü

MDWT (Multi-level Discrete Wavelet Transform) + TSK-LHP özellik pipeline'ı.
TSK-LHP: Ternary Signum Key - Local Histogram Pattern

NumPy vektörizasyonu ile MATLAB versiyonundan çok daha hızlı.
"""

import numpy as np
import pywt
import soundfile as sf

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

from config import NUM_LEVELS, WAVELET, NUM_BINS, SAMPLE_RATE


def tsk_lhp(signal: np.ndarray, num_bins: int = NUM_BINS) -> np.ndarray:
    """
    TSK-LHP özellik çıkarma — tam vektörize implementasyon.

    Adımlar:
      1. Sinyal 16'lık kayan pencerelere bölünür (örtüşen).
      2. Her pencere iki yarıya ayrılır (ilk 8 vs son 8).
      3. Fark vektöründen signum ve ternary bitleri hesaplanır.
      4. 8 bit → 0-255 arası ondalık dönüşümü.
      5. 3 histogram (signum, upper ternary, lower ternary) çıkarılır.
      6. Histogramlar normalize edilip birleştirilir.

    Args:
        signal  : 1-D float64 sinyal (DWT approx katsayıları)
        num_bins: Histogram bin sayısı (256 = 8-bit)

    Returns:
        (num_bins * 3,) float32 özellik vektörü
    """
    window_size = 16
    n = len(signal)

    if n < window_size:
        return np.zeros(num_bins * 3, dtype=np.float32)

    # Standart sapmanın yarısı = ternary eşiği
    thresh = float(np.std(signal)) / 2.0
    if thresh < 1e-12:
        thresh = 1e-12

    # ── Kayan pencere matrisi: (num_windows, 16) ─────────────────────────────
    num_windows = n - window_size + 1
    row_idx = np.arange(num_windows)[:, None]     # (N_w, 1)
    col_idx = np.arange(window_size)[None, :]     # (1, 16)
    windows = signal[row_idx + col_idx]            # (N_w, 16) — views, no copy

    o    = windows[:, :8]   # İlk 8 eleman
    d    = windows[:, 8:]   # Son 8 eleman
    diff = o - d             # (N_w, 8)

    # ── Bit hesaplama ─────────────────────────────────────────────────────────
    signum_bits = (diff >= 0).astype(np.float32)               # 0 veya 1
    upper_bits  = (diff >  thresh).astype(np.float32)          # üst ternary
    lower_bits  = (diff < -thresh).astype(np.float32)          # alt ternary

    # ── Bit → ondalık (LSB önce: bit0=1, bit1=2, bit2=4, ...) ───────────────
    powers = (2 ** np.arange(8)).astype(np.float32)            # [1,2,4,...,128]
    sig_vals   = signum_bits @ powers   # (N_w,)
    upper_vals = upper_bits  @ powers
    lower_vals = lower_bits  @ powers

    # ── Normalize histogram ───────────────────────────────────────────────────
    bins = np.arange(num_bins + 1, dtype=np.float32)

    h_s = np.histogram(sig_vals,   bins=bins)[0].astype(np.float32)
    h_u = np.histogram(upper_vals, bins=bins)[0].astype(np.float32)
    h_l = np.histogram(lower_vals, bins=bins)[0].astype(np.float32)

    for h in (h_s, h_u, h_l):
        total = h.sum()
        if total > 0:
            h /= total

    return np.concatenate([h_s, h_u, h_l])   # 768 boyutlu


def extract_features(
    audio: np.ndarray,
    num_levels: int = NUM_LEVELS,
    wavelet: str = WAVELET,
) -> np.ndarray | None:
    """
    Bir ses sinyalinden MDWT + TSK-LHP özellik vektörü çıkarır.

    Her DWT seviyesinde approx. katsayılarına TSK-LHP uygulanır.
    Toplam boyut: num_levels * NUM_BINS * 3 = 7 * 768 = 5376

    Args:
        audio      : Normalize edilmemiş ham ses sinyali (1-D float)
        num_levels : DWT seviye sayısı
        wavelet    : PyWavelets dalgacık adı

    Returns:
        (5376,) özellik vektörü, sinyal çok kısa ise None
    """
    if len(audio) < 2 ** num_levels:
        return None

    # Genlişik normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio.astype(np.float64) / peak

    all_feats = []
    for level in range(1, num_levels + 1):
        # Her seviye için bağımsız wavedec — approx katsayıları direkt alınır
        # (MATLAB appcoef() muadili)
        coeffs = pywt.wavedec(audio, wavelet, level=level)
        approx = coeffs[0]   # Approximation coefficients at this level
        all_feats.append(tsk_lhp(approx.astype(np.float64)))

    return np.concatenate(all_feats).astype(np.float32)   # 5376 boyutlu


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Ses dosyasını yükler, mono'ya çevirir ve hedef Hz'e yeniden örnekler.
    FLAC, WAV, OGG, MP3 vb. desteklenir.

    Args:
        file_path : Ses dosyasının tam yolu
        target_sr : Hedef örnekleme frekansı (Hz)

    Returns:
        (audio_array, sample_rate) çifti
    """
    audio, sr = sf.read(file_path, always_2d=False)

    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)

    # Yeniden örnekleme
    if sr != target_sr:
        if _HAS_LIBROSA:
            audio = librosa.resample(
                audio.astype(np.float32), orig_sr=sr, target_sr=target_sr
            ).astype(np.float64)
        else:
            # Librosa yoksa doğrusal interpolasyon (yeterli kalite)
            new_len = int(len(audio) * target_sr / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            )

    return audio, target_sr
