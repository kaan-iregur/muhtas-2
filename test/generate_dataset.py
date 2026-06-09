"""
Gerçekçi IHA Ses Veri Seti Üretici
===================================

Önceki versiyonun problemleri:
  1. BPF değerleri sınıflar arası çok farklıydı (trivially separable)
  2. Gürültü seviyesi gerçek dışı düşüktü
  3. Sabit devir — gerçekte RPM uçuş boyunca dalgalanır
  4. Arka plan gürültüsü yoktu

Bu versiyon:
  - Birbirine yakın BPF'lere sahip gerçekçi IHA tipleri (DJI Mavic 3 ≈ DJI Air 2S)
  - Uçuş boyunca RPM drift simülasyonu
  - Ortam gürültüsü (rüzgar, pembe gürültü) — SNR 5–20 dB
  - Mesafe ve yön değişimi simülasyonu (genlik zarfı)
  - Doğrusal olmayan harmonik bozunma
"""

import os
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt, sosfilt, butter as sos_butter

# ─── Gerçekçi IHA Konfigürasyonları ──────────────────────────────────────────
#
# Kaynak: Gerçek IHA ölçümlerinden alınan tipik değer aralıkları
#
# DJI Mavic 3     : 4 motor, ~4600 RPM, 2 kanat → BPF ≈ 153 Hz
# DJI Air 2S      : 4 motor, ~5100 RPM, 2 kanat → BPF ≈ 170 Hz  (Mavic'e yakın!)
# DJI Matrice 300 : 6 motor, ~3200 RPM, 2 kanat → BPF ≈ 107 Hz
# Parrot Anafi    : 4 motor, ~5500 RPM, 2 kanat → BPF ≈ 183 Hz
# Sabit kanat     : 1 motor, ~7000 RPM, 2 kanat → BPF ≈ 233 Hz
#
# Dikkat: Mavic 3 ve Air 2S BPF'leri birbirine çok yakın (153 vs 170 Hz)
# Bu, gerçek dünyada karşılaşılan zorlukların ta kendisi.
# ─────────────────────────────────────────────────────────────────────────────

DRONE_CONFIGS = {
    # 4 motorlu küçük tüketici drone — yüksek devir, küçük pervane
    "DJI_Mavic3": {
        "num_motors":      4,
        "rpm_mean":        4600,
        "rpm_std":         350,      # Uçuş boyunca ±350 RPM dalgalanma
        "rpm_drift_hz":    0.8,      # Devir dalgalanma frekansı (Hz)
        "rpm_drift_depth": 0.04,     # ±%4 devir değişimi
        "num_blades":      2,
        "harmonic_count":  8,
        "harmonic_decay":  0.60,
        "bpf_harmonics":   5,
        "bpf_amp":         0.40,
        "aero_noise":      0.25,     # Gerçekçi: pervane türbülansı
        "elec_noise":      0.08,
        "mod_freq_hz":     7.0,
        "mod_depth":       0.20,
        "distance_m":      (15, 80),
        "nonlinear_coef":  0.05,     # Harmonik bozunma katsayısı
    },
    # 4 motorlu orta sınıf drone — Mavic'e benzer ama farklı
    "DJI_Air2S": {
        "num_motors":      4,
        "rpm_mean":        5100,
        "rpm_std":         400,
        "rpm_drift_hz":    1.1,
        "rpm_drift_depth": 0.05,
        "num_blades":      2,
        "harmonic_count":  8,
        "harmonic_decay":  0.58,
        "bpf_harmonics":   5,
        "bpf_amp":         0.42,
        "aero_noise":      0.28,
        "elec_noise":      0.09,
        "mod_freq_hz":     8.5,
        "mod_depth":       0.22,
        "distance_m":      (10, 60),
        "nonlinear_coef":  0.06,
    },
    # 6 motorlu endüstriyel — ağır, yavaş
    "DJI_Matrice": {
        "num_motors":      6,
        "rpm_mean":        3200,
        "rpm_std":         220,
        "rpm_drift_hz":    0.5,
        "rpm_drift_depth": 0.03,
        "num_blades":      2,
        "harmonic_count":  10,
        "harmonic_decay":  0.70,
        "bpf_harmonics":   6,
        "bpf_amp":         0.50,
        "aero_noise":      0.35,     # 6 motor = daha fazla türbülans
        "elec_noise":      0.10,
        "mod_freq_hz":     4.5,
        "mod_depth":       0.15,
        "distance_m":      (10, 50),
        "nonlinear_coef":  0.04,
    },
    # 4 motorlu kompakt — Parrot Anafi tipi, hafif gövde
    "Parrot_Anafi": {
        "num_motors":      4,
        "rpm_mean":        5500,
        "rpm_std":         500,      # Kompakt gövde = daha fazla titreşim
        "rpm_drift_hz":    1.3,
        "rpm_drift_depth": 0.06,
        "num_blades":      2,
        "harmonic_count":  7,
        "harmonic_decay":  0.55,
        "bpf_harmonics":   4,
        "bpf_amp":         0.38,
        "aero_noise":      0.30,
        "elec_noise":      0.12,
        "mod_freq_hz":     10.0,
        "mod_depth":       0.25,
        "distance_m":      (5, 40),
        "nonlinear_coef":  0.08,
    },
    # Sabit kanat — tek motor, yüksek devir, çok farklı ses karakteri
    "Sabit_Kanat": {
        "num_motors":      1,
        "rpm_mean":        7000,
        "rpm_std":         600,
        "rpm_drift_hz":    0.3,      # Sabit kanat daha kararlı RPM
        "rpm_drift_depth": 0.02,
        "num_blades":      2,
        "harmonic_count":  14,       # Çok zengin harmonik içerik
        "harmonic_decay":  0.78,
        "bpf_harmonics":   7,
        "bpf_amp":         0.55,
        "aero_noise":      0.15,     # Daha az türbülans (sabit kanat)
        "elec_noise":      0.04,
        "mod_freq_hz":     2.0,      # Çok yavaş — kararlı motor
        "mod_depth":       0.08,
        "distance_m":      (30, 150),
        "nonlinear_coef":  0.03,
    },
}


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pembe gürültü üret (1/f spektrumu — gerçekçi ortam gürültüsü)."""
    white  = rng.standard_normal(n)
    fft    = np.fft.rfft(white)
    freqs  = np.fft.rfftfreq(n)
    freqs[0] = 1e-10
    fft   /= np.sqrt(freqs)
    pink   = np.fft.irfft(fft, n)
    return pink / (np.std(pink) + 1e-10)


def _wind_noise(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Rüzgar gürültüsü: düşük frekanslı pembe gürültü."""
    raw = _pink_noise(n, rng)
    b, a = butter(4, 800 / (sr / 2), btype="low")
    return filtfilt(b, a, raw)


def generate_drone_audio(
    sr:       int,
    duration: float,
    cfg:      dict,
    seed:     int | None = None,
    snr_db:   float | None = None,   # None = rastgele 5–20 dB
) -> np.ndarray:
    """
    Gerçekçi IHA ses sinyali üretir.

    Özellikler:
      - Uçuş boyunca RPM drift (non-stationary sinyal)
      - Motor başına hafif devir farkı (balans hatası)
      - Doğrusal olmayan harmonik bozunma
      - Pembe gürültü + rüzgar arka planı (ayarlanabilir SNR)
      - Genlik zarfı (IHA yaklaşıyor/uzaklaşıyor simülasyonu)
      - Mikrofon bandpass filtresi (50 Hz – 7500 Hz)
    """
    rng = np.random.default_rng(seed)
    n   = int(sr * duration)
    t   = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros(n, dtype=np.float64)

    # ── RPM drift: uçuş boyunca devir yavaşça değişir ─────────────────────
    drift_hz    = cfg["rpm_drift_hz"]
    drift_depth = cfg["rpm_drift_depth"]
    drift_phase = rng.uniform(0, 2 * np.pi)
    # Anlık RPM = rpm_mean × (1 + drift_depth × sin(2π × drift_hz × t))
    rpm_envelope = cfg["rpm_mean"] * (
        1.0 + drift_depth * np.sin(2 * np.pi * drift_hz * t + drift_phase)
    )

    num_motors    = cfg["num_motors"]
    num_blades    = cfg["num_blades"]
    harmonic_count= cfg["harmonic_count"]
    harmonic_decay= cfg["harmonic_decay"]
    bpf_harmonics = cfg["bpf_harmonics"]
    bpf_amp       = cfg["bpf_amp"]

    # ── Her motor: biraz farklı anlık devir ───────────────────────────────
    for m in range(num_motors):
        offset = rng.normal(0, cfg["rpm_std"]) / cfg["rpm_mean"]
        rpm_m  = rpm_envelope * (1.0 + offset)
        rpm_m  = np.maximum(rpm_m, cfg["rpm_mean"] * 0.6)

        # Anlık faz = ∫ 2π × f(t) dt  (kümülatif toplam)
        f_motor = rpm_m / 60.0
        phase_m = 2 * np.pi * np.cumsum(f_motor) / sr
        phase_m += 2 * np.pi * m / num_motors  # Motor faz ofseti

        motor_amp = 1.0 / num_motors

        # Motor harmonikleri
        for n_h in range(1, harmonic_count + 1):
            amp = motor_amp * (harmonic_decay ** (n_h - 1))
            sig += amp * np.sin(n_h * phase_m)

        # BPF harmonikleri
        f_bpf     = num_blades * rpm_m / 60.0
        phase_bpf = 2 * np.pi * np.cumsum(f_bpf) / sr
        for n_b in range(1, bpf_harmonics + 1):
            amp = bpf_amp / num_motors * (0.65 ** (n_b - 1))
            sig += amp * np.sin(n_b * phase_bpf)

    # ── Doğrusal olmayan harmonik bozunma (motor saturasyonu) ─────────────
    coef = cfg.get("nonlinear_coef", 0.0)
    if coef > 0:
        sig += coef * np.sign(sig) * sig ** 2

    # ── Genlik modülasyonu ────────────────────────────────────────────────
    mod_f = cfg["mod_freq_hz"] * rng.uniform(0.7, 1.3)
    mod_d = cfg["mod_depth"]   * rng.uniform(0.6, 1.4)
    sig  *= 1.0 + mod_d * np.sin(2 * np.pi * mod_f * t + rng.uniform(0, 2 * np.pi))

    # ── Mesafe + yön zarfı: IHA yaklaşıp uzaklaşıyor ─────────────────────
    d_min, d_max = cfg["distance_m"]
    d_start = rng.uniform(d_min, d_max)
    d_end   = rng.uniform(d_min, d_max)
    dist_env = np.linspace(d_start, d_end, n)
    sig     *= d_min / dist_env        # 1/r zayıflama

    # ── Aerodinamik türbülans (band-pass filtrelenmiş) ────────────────────
    aero = rng.standard_normal(n) * cfg["aero_noise"]
    b, a  = butter(3, [80 / (sr / 2), 5000 / (sr / 2)], btype="band")
    aero  = filtfilt(b, a, aero)
    sig  += aero

    # ── Elektrik gürültüsü ────────────────────────────────────────────────
    sig += rng.standard_normal(n) * cfg["elec_noise"]

    # ── Sinyal gücünü normalize et (SNR hesabı için) ──────────────────────
    sig_power = np.mean(sig ** 2) + 1e-12
    sig /= np.sqrt(sig_power)   # Birim güç

    # ── Arka plan gürültüsü (SNR kontrollü) ──────────────────────────────
    if snr_db is None:
        snr_db = rng.uniform(5.0, 20.0)   # 5–20 dB arası rastgele

    snr_linear = 10 ** (snr_db / 10.0)

    # Pembe gürültü + rüzgar karışımı
    pink   = _pink_noise(n, rng)
    wind   = _wind_noise(n, sr, rng)
    noise  = 0.7 * pink + 0.3 * wind
    noise /= (np.sqrt(np.mean(noise ** 2)) + 1e-12)  # Birim güç

    noise_amp = np.sqrt(1.0 / snr_linear)
    sig      += noise_amp * noise

    # ── Mikrofon bandpass filtresi (50 Hz – 7500 Hz) ──────────────────────
    b_mic, a_mic = butter(2, [50 / (sr / 2), min(7500 / (sr / 2), 0.99)], btype="band")
    sig = filtfilt(b_mic, a_mic, sig)

    # ── Son normalize ─────────────────────────────────────────────────────
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig /= peak

    return sig.astype(np.float64)


def generate_dataset(
    output_dir:  str   = "dataset",
    samples:     int   = 50,
    duration:    float = 3.0,
    sr:          int   = 16000,
    drone_types: list  = None,
    base_seed:   int   = 42,
    snr_db:      float | None = None,   # None = rastgele 5–20 dB
    snr_label:   str   = "",            # Klasör adı eki (örn. "_snr10")
) -> dict:
    """
    Her IHA tipi için 'samples' adet FLAC dosyası üretir.
    """
    if drone_types is None:
        drone_types = list(DRONE_CONFIGS.keys())

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" SENTETİK VERİ ÜRETİCİ  (Gerçekçi mod)")
    print(f"{'='*60}")
    print(f"  Sınıf sayısı  : {len(drone_types)}")
    print(f"  Sınıf başına  : {samples} dosya")
    print(f"  Süre          : {duration} s")
    print(f"  SR            : {sr} Hz")
    print(f"  SNR           : {'5–20 dB rastgele' if snr_db is None else f'{snr_db} dB'}")
    print(f"  Tohum         : {base_seed}")
    print()

    # BPF bilgisi (ayırt edilebilirlik uyarısı)
    bpfs = {}
    for name in drone_types:
        if name in DRONE_CONFIGS:
            cfg  = DRONE_CONFIGS[name]
            bpf  = cfg["num_blades"] * cfg["rpm_mean"] / 60
            bpfs[name] = bpf

    print("  BPF değerleri (birbirine yakın = daha zor):")
    for name, bpf in sorted(bpfs.items(), key=lambda x: x[1]):
        print(f"    {name:<20}: {bpf:>6.1f} Hz")
    print()

    summary = {}
    total   = len(drone_types) * samples
    done    = 0

    for cls_name in drone_types:
        cfg      = DRONE_CONFIGS[cls_name]
        cls_dir  = os.path.join(output_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        bpf = cfg["num_blades"] * cfg["rpm_mean"] / 60
        print(f"  [{cls_name}]  BPF={bpf:.0f} Hz  |  "
              f"{cfg['num_motors']} motor  |  aero_noise={cfg['aero_noise']}")

        for i in range(samples):
            seed  = base_seed + abs(hash(cls_name)) % 100000 + i * 7
            audio = generate_drone_audio(sr, duration, cfg,
                                         seed=seed, snr_db=snr_db)
            fname = f"{cls_name}_{i:04d}.flac"
            sf.write(os.path.join(cls_dir, fname), audio, sr,
                     format="flac", subtype="PCM_16")
            done += 1
            pct  = done / total * 100
            bar  = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
            print(f"\r    [{bar}] {pct:5.1f}%", end="", flush=True)

        print(f"\r    [{'█'*33}] 100.0%  — {samples} dosya        ")
        summary[cls_name] = samples

    print(f"\n  Toplam: {total} dosya  →  {output_dir}")
    print(f"{'='*60}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",   default="dataset")
    parser.add_argument("--samples",  type=int,   default=50)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--sr",       type=int,   default=16000)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--snr",      type=float, default=None)
    args = parser.parse_args()
    generate_dataset(args.output, args.samples, args.duration,
                     args.sr, base_seed=args.seed, snr_db=args.snr)
