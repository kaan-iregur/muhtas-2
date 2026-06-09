"""
IHA Ses Sınıflandırma — Bağımsız Test Platformu (Dürüst Versiyon)
==================================================================

Önceki versiyonun problemi:
  - BPF değerleri sınıflar arası çok farklıydı (25 Hz vs 293 Hz)
  - Gürültü gerçek dışı düşüktü → trivially separable → %100 çıkması anlamsızdı

Bu versiyon:
  - Birbirine yakın BPF'ler (107–233 Hz arası 5 sınıf)
  - Gerçekçi SNR: eğitim 5–20 dB, test de 5–20 dB
  - RPM drift (non-stationary)
  - SNR stres testi: 20 dB → 10 dB → 5 dB → 0 dB
  - Train/test tamamen farklı tohumlar

Kullanım:
  python run_test.py                  # Tam pipeline
  python run_test.py --skip-generate  # Veri zaten üretildiyse
  python run_test.py --only-eval      # Sadece eval (model eğitildiyse)
  python run_test.py --sim-seconds 40
"""

import os
import sys
import glob
import time
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

# ─── Yollar ──────────────────────────────────────────────────────────────────
TEST_DIR  = Path(__file__).parent.resolve()
MAIN_DIR  = TEST_DIR.parent / "iha_sistemi"
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(TEST_DIR))

from generate_dataset import generate_dataset, generate_drone_audio, DRONE_CONFIGS
from feature_extractor import extract_features, load_audio
from config import SAMPLE_RATE, SEGMENT_DURATION, NUM_LEVELS, WAVELET, HOP_DURATION

TRAIN_DIR   = str(TEST_DIR / "dataset_train")
TEST_DIR_D  = str(TEST_DIR / "dataset_test")
MODEL_DIR   = str(TEST_DIR / "models")
RESULTS_DIR = str(TEST_DIR / "sonuclar")

# ─── Farklı tohumlar — kritik ─────────────────────────────────────────────────
SEED_TRAIN = 42
SEED_TEST  = 91273    # Tamamen farklı tohum

SAMPLES    = 60       # Sınıf başına dosya
DURATION   = 3.0      # Saniye


# ══════════════════════════════════════════════════════════════════════════════
#  1. VERİ ÜRETİMİ
# ══════════════════════════════════════════════════════════════════════════════

def step_generate():
    print("\n" + "═"*62)
    print("  ADIM 1 — VERİ ÜRETİMİ")
    print("  Eğitim ve test FARKLI tohumlar, FARKLI SNR örneklemeleri")
    print("═"*62)

    print(f"\n  [EĞİTİM]  tohum={SEED_TRAIN}, SNR=5–20 dB rastgele")
    generate_dataset(TRAIN_DIR, SAMPLES, DURATION, SAMPLE_RATE,
                     base_seed=SEED_TRAIN, snr_db=None)

    print(f"\n  [TEST]  tohum={SEED_TEST}, SNR=5–20 dB rastgele")
    print("  (Model bu sesleri eğitimde hiç görmeyecek)\n")
    generate_dataset(TEST_DIR_D, SAMPLES, DURATION, SAMPLE_RATE,
                     base_seed=SEED_TEST, snr_db=None)


# ══════════════════════════════════════════════════════════════════════════════
#  2. EĞİTİM
# ══════════════════════════════════════════════════════════════════════════════

def step_train():
    print("\n" + "═"*62)
    print("  ADIM 2 — MODEL EĞİTİMİ  (yalnızca eğitim verisi)")
    print("═"*62)
    from train import train as run_train
    run_train(dataset_dir=TRAIN_DIR, model_dir=MODEL_DIR,
              excel_dir=RESULTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
#  3. BAĞIMSIZ TEST
# ══════════════════════════════════════════════════════════════════════════════

def _extract_dir(data_dir: str):
    classes = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))])
    seg_len = int(SAMPLE_RATE * SEGMENT_DURATION)
    X, y = [], []
    for ci, cn in enumerate(classes):
        for fp in glob.glob(os.path.join(data_dir, cn, "*.flac")):
            try:
                audio, _ = load_audio(fp, SAMPLE_RATE)
                for s in range(0, max(1, len(audio)-seg_len+1), seg_len):
                    f = extract_features(audio[s:s+seg_len])
                    if f is not None:
                        X.append(f); y.append(ci)
            except Exception as e:
                print(f"    HATA: {e}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), classes


def step_evaluate(model_pkg: dict):
    print("\n" + "═"*62)
    print("  ADIM 3 — BAĞIMSIZ TEST  (model hiç görmedi)")
    print("═"*62)

    print("\n  Özellikler çıkarılıyor...")
    X, y, classes = _extract_dir(TEST_DIR_D)
    print(f"  {len(X)} segment  |  sınıflar: {classes}")

    tr     = model_pkg["transformer"]
    scaler = model_pkg["scaler"]
    clf    = model_pkg["classifier"]

    X_sel  = tr.transform(X.astype(np.float64))
    X_sc   = scaler.transform(X_sel)

    t0     = time.perf_counter()
    y_pred = clf.predict(X_sc)
    lat    = (time.perf_counter() - t0) / len(X) * 1000

    acc  = accuracy_score(y, y_pred)
    cm   = confusion_matrix(y, y_pred)
    rep  = classification_report(y, y_pred, target_names=classes, digits=3)
    p, r, f1, _ = precision_recall_fscore_support(y, y_pred, average="weighted")

    print(f"\n  ┌{'─'*42}┐")
    print(f"  │  Doğruluk   : %{acc*100:6.2f}                    │")
    print(f"  │  Kesinlik   : %{p*100:6.2f}  (ağırlıklı)      │")
    print(f"  │  Duyarlılık : %{r*100:6.2f}  (ağırlıklı)      │")
    print(f"  │  F1 Skoru   : %{f1*100:6.2f}  (ağırlıklı)      │")
    print(f"  │  Gecikme    : {lat:6.3f} ms / örnek           │")
    print(f"  └{'─'*42}┘")
    print(f"\n  Sınıflandırma Raporu:\n  {'─'*46}")
    for ln in rep.split("\n"):
        print(f"  {ln}")

    return dict(y_test=y, y_pred=y_pred, accuracy=acc,
                precision=p, recall=r, f1=f1,
                cm=cm, report=rep, lat_ms=lat, classes=classes)


# ══════════════════════════════════════════════════════════════════════════════
#  4. SNR STRES TESTİ — Gerçek dünya dayanıklılık analizi
# ══════════════════════════════════════════════════════════════════════════════

def step_snr_stress(model_pkg: dict, snr_levels=(20, 15, 10, 5, 0)):
    """
    Modeli farklı SNR seviyelerinde test eder.
    Gerçek dünyada mesafeye / ortama göre SNR değişir.
    Bu test, sistemin hangi gürültü seviyesine kadar çalıştığını gösterir.
    """
    print("\n" + "═"*62)
    print("  ADIM 4 — SNR STRES TESTİ")
    print("  (Her SNR seviyesinde 30 farklı ses üretilip test edilir)")
    print("═"*62)

    classes  = model_pkg["classes"]
    tr       = model_pkg["transformer"]
    scaler   = model_pkg["scaler"]
    clf      = model_pkg["classifier"]

    results  = {}
    seg_len  = int(SAMPLE_RATE * SEGMENT_DURATION)
    n_per    = 30   # Her sınıf/SNR başına örnek

    print(f"\n  {'SNR':>6}  {'Doğruluk':>10}  Sınıf bazında doğruluk")
    print(f"  {'─'*62}")

    for snr in snr_levels:
        X_snr, y_snr = [], []

        for ci, cn in enumerate(classes):
            if cn not in DRONE_CONFIGS:
                continue
            cfg = DRONE_CONFIGS[cn]
            for i in range(n_per):
                seed  = 77777 + ci * 1000 + i * 13 + int(snr * 100)
                audio = generate_drone_audio(
                    SAMPLE_RATE, DURATION, cfg, seed=seed, snr_db=float(snr)
                )
                feat = extract_features(audio[:seg_len])
                if feat is not None:
                    X_snr.append(feat); y_snr.append(ci)

        X_snr = np.array(X_snr, dtype=np.float64)
        y_snr = np.array(y_snr, dtype=np.int32)

        X_sel  = tr.transform(X_snr)
        X_sc   = scaler.transform(X_sel)
        y_pred = clf.predict(X_sc)
        acc    = accuracy_score(y_snr, y_pred)

        # Sınıf bazında doğruluk
        per_cls = []
        for ci in range(len(classes)):
            mask = y_snr == ci
            if mask.sum() > 0:
                per_cls.append(f"{accuracy_score(y_snr[mask], y_pred[mask])*100:.0f}%")

        results[snr] = {"accuracy": acc, "per_class": per_cls,
                        "cm": confusion_matrix(y_snr, y_pred)}

        cls_str = "  ".join(f"{c:>5}" for c in per_cls)
        print(f"  {snr:>4} dB  {acc*100:>9.2f}%  [{cls_str}]")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  5. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

CMAP = LinearSegmentedColormap.from_list("iha", ["#ffffff", "#1a6bb5", "#0a3060"])


def _plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=CMAP)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=classes, yticklabels=classes,
           ylabel="Gerçek", xlabel="Tahmin", title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            v   = cm[i, j]
            pct = v / max(cm[i].sum(), 1) * 100
            ax.text(j, i, f"{v}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=8, color="white" if v > thresh else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def _plot_snr_curve(snr_results: dict, classes: list, path: str):
    snrs = sorted(snr_results.keys(), reverse=True)
    accs = [snr_results[s]["accuracy"] * 100 for s in snrs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: genel doğruluk vs SNR
    ax1.plot(snrs, accs, "o-", lw=2.5, color="#1565C0", markersize=9)
    for snr, acc in zip(snrs, accs):
        ax1.annotate(f"%{acc:.1f}", (snr, acc),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10, fontweight="bold")
    ax1.set_xlabel("SNR (dB)", fontsize=12)
    ax1.set_ylabel("Doğruluk (%)", fontsize=12)
    ax1.set_title("Doğruluk vs Gürültü Seviyesi\n"
                  "(Düşük dB = daha gürültülü ortam)", fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.axhline(50, ls="--", color="red",    lw=1, label="Rastgele eşiği (%20 = 5 sınıf için)")
    ax1.axhline(80, ls="--", color="orange", lw=1, label="Kabul edilebilir (%80)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Sağ: en zor SNR'da karmaşıklık matrisi
    worst_snr = min(snrs)
    worst_cm  = snr_results[worst_snr]["cm"]
    im = ax2.imshow(worst_cm, cmap=CMAP)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    ax2.set(xticks=ticks, yticks=ticks,
            xticklabels=classes, yticklabels=classes,
            title=f"En Zor Durum — SNR={worst_snr} dB\n"
                  f"(Doğruluk: %{snr_results[worst_snr]['accuracy']*100:.1f})",
            ylabel="Gerçek", xlabel="Tahmin")
    plt.setp(ax2.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax2.get_yticklabels(), fontsize=8)
    thresh = worst_cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = worst_cm[i, j]
            ax2.text(j, i, str(v), ha="center", va="center",
                     fontsize=9, color="white" if v > thresh else "black")

    fig.suptitle("SNR Stres Testi — Sistem Gürültüye Dayanıklılık Analizi",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def _plot_bpf_comparison(path: str):
    """BPF değerlerini ve çakışmayı görsel olarak göster."""
    configs = list(DRONE_CONFIGS.items())
    n = len(configs)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 7))

    for col, (name, cfg) in enumerate(configs):
        sr   = SAMPLE_RATE
        dur  = 1.5
        a_tr = generate_drone_audio(sr, dur, cfg, seed=SEED_TRAIN+col, snr_db=15)
        a_te = generate_drone_audio(sr, dur, cfg, seed=SEED_TEST+col,  snr_db=15)

        t = np.linspace(0, dur, len(a_tr))

        ax = axes[0, col]
        ax.plot(t[:sr//8], a_tr[:sr//8], lw=0.6, color="#1565C0", alpha=0.7, label="Eğitim")
        ax.plot(t[:sr//8], a_te[:sr//8], lw=0.6, color="#E53935", alpha=0.7, label="Test")
        ax.set_title(name.replace("_", "\n"), fontsize=9, fontweight="bold")
        ax.set_ylim(-1.1, 1.1); ax.tick_params(labelsize=6); ax.grid(alpha=0.3)
        ax.legend(fontsize=6)

        ax2 = axes[1, col]
        for audio, color in [(a_tr, "#1565C0"), (a_te, "#E53935")]:
            fft  = np.abs(np.fft.rfft(audio))**2
            freq = np.fft.rfftfreq(len(audio), 1/sr)
            ax2.plot(freq[freq<=3000], 10*np.log10(fft[freq<=3000]+1e-12),
                     lw=0.7, color=color, alpha=0.8)
        bpf = cfg["num_blades"] * cfg["rpm_mean"] / 60
        ax2.axvline(bpf, color="gold", lw=1.5, ls="--", label=f"BPF={bpf:.0f} Hz")
        ax2.set_xlim(0, 3000); ax2.grid(alpha=0.3)
        ax2.legend(fontsize=6, loc="upper right")
        ax2.tick_params(labelsize=6)

    fig.suptitle("Eğitim (mavi) vs Test (kırmızı) — SNR=15 dB\n"
                 "BPF değerleri artık birbirine daha yakın → daha zor sınıflandırma",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def _plot_metrics_bar(classes, y_test, y_pred, path):
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=range(len(classes)))
    x, w = np.arange(len(classes)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-w, p_c,  w, label="Kesinlik",   color="#2196F3", alpha=0.85)
    ax.bar(x,   r_c,  w, label="Duyarlılık", color="#4CAF50", alpha=0.85)
    ax.bar(x+w, f1_c, w, label="F1 Skoru",   color="#FF5722", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Skor")
    ax.set_title("Sınıf Bazında Performans — Bağımsız Test Verisi\n"
                 "(Gerçekçi gürültü ve birbirine yakın BPF'ler)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in ax.patches:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


def step_visualize(eval_res, model_pkg, snr_results):
    print("\n" + "═"*62)
    print("  ADIM 5 — GÖRSELLEŞTİRME")
    print("═"*62)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    classes = eval_res["classes"]
    acc     = eval_res["accuracy"]

    _plot_cm(eval_res["cm"], classes,
             f"Karmaşıklık Matrisi — Bağımsız Test Verisi\n"
             f"(Gerçekçi SNR 5–20 dB, model hiç görmedi)\n"
             f"Doğruluk: %{acc*100:.2f}",
             os.path.join(RESULTS_DIR, "1_karmasiklik_matrisi.png"))

    _plot_metrics_bar(classes, eval_res["y_test"], eval_res["y_pred"],
                      os.path.join(RESULTS_DIR, "2_sinif_metrikleri.png"))

    _plot_snr_curve(snr_results, classes,
                    os.path.join(RESULTS_DIR, "3_snr_stres_testi.png"))

    _plot_bpf_comparison(
        os.path.join(RESULTS_DIR, "4_ses_spektrumlari.png"))

    # Metin raporu
    rp = os.path.join(RESULTS_DIR, "rapor.txt")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("IHA SES SINIFLANDIRMA — DÜRÜST TEST RAPORU\n")
        f.write("=" * 52 + "\n\n")
        f.write("VERİ KOŞULLARI\n")
        f.write(f"  Eğitim tohumu  : {SEED_TRAIN}\n")
        f.write(f"  Test tohumu    : {SEED_TEST}\n")
        f.write(f"  SNR (eğitim)   : 5–20 dB rastgele\n")
        f.write(f"  SNR (test)     : 5–20 dB rastgele\n")
        f.write(f"  BPF aralığı    : 107–233 Hz\n\n")
        f.write("GENEL SONUÇLAR\n")
        f.write(f"  Eğitim CV      : %{model_pkg['best_accuracy']*100:.2f}\n")
        f.write(f"  Test Doğruluk  : %{acc*100:.2f}\n")
        f.write(f"  Test F1        : %{eval_res['f1']*100:.2f}\n")
        f.write(f"  Gecikme        : {eval_res['lat_ms']:.3f} ms/örnek\n\n")
        f.write("SNR STRES TESTİ\n")
        for snr, res in sorted(snr_results.items(), reverse=True):
            f.write(f"  SNR={snr:>3} dB  →  %{res['accuracy']*100:.2f}\n")
        f.write("\nSINIFLANDIRMA RAPORU\n")
        f.write("-" * 52 + "\n")
        f.write(eval_res["report"])
    print(f"  → {rp}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. GERÇEK ZAMANLI SİMÜLASYON
# ══════════════════════════════════════════════════════════════════════════════

def step_simulate(model_pkg: dict, total_seconds: int = 30):
    print("\n" + "═"*62)
    print(f"  ADIM 6 — GERÇEK ZAMANLI SİMÜLASYON  ({total_seconds}s)")
    print("  (Test tohumu + rastgele SNR — model hiç görmedi)")
    print("═"*62)

    classes = model_pkg["classes"]
    tr      = model_pkg["transformer"]
    scaler  = model_pkg["scaler"]
    clf     = model_pkg["classifier"]

    hop_len = int(SAMPLE_RATE * HOP_DURATION)
    seg_len = int(SAMPLE_RATE * SEGMENT_DURATION)
    switch  = 5
    seq     = [classes[i % len(classes)]
               for i in range(total_seconds // switch + 2)]

    buffer  = np.zeros(seg_len * 2, dtype=np.float64)
    buf_ptr = 0
    correct = total_pred = 0
    log     = []

    print(f"\n  {'Zaman':>6}  {'Gerçek':^20}  {'Tahmin':^20}  {'Güven':>6}  {'SNR':>5}  OK?")
    print(f"  {'─'*66}")

    for hop in range(total_seconds * 2):
        t_now  = hop * HOP_DURATION
        slot   = min(int(t_now / switch), len(seq) - 1)
        true_c = seq[slot]
        if true_c not in DRONE_CONFIGS:
            continue

        # Rastgele SNR — gerçek dünya gibi
        rng_hop = np.random.default_rng(SEED_TEST + hop * 53)
        snr     = float(rng_hop.uniform(5, 20))
        cfg     = DRONE_CONFIGS[true_c]
        chunk   = generate_drone_audio(
            SAMPLE_RATE, HOP_DURATION, cfg,
            seed=SEED_TEST + hop * 97, snr_db=snr
        )

        buffer   = np.roll(buffer, -hop_len)
        buffer[-hop_len:] = chunk[:hop_len]
        buf_ptr += hop_len

        if buf_ptr < seg_len:
            continue

        feat = extract_features(buffer[-seg_len:], NUM_LEVELS, WAVELET)
        if feat is None:
            continue

        X_sel  = tr.transform(feat.reshape(1,-1).astype(np.float64))
        X_sc   = scaler.transform(X_sel)
        t_inf  = time.perf_counter()
        pi     = int(clf.predict(X_sc)[0])
        lat    = (time.perf_counter() - t_inf) * 1000
        pred_c = classes[pi]

        conf = (float(clf.predict_proba(X_sc)[0][pi])
                if hasattr(clf, "predict_proba") else 1.0)

        ok = pred_c == true_c
        correct += int(ok); total_pred += 1

        G, R = ("\033[92m", "\033[0m") if ok else ("\033[91m", "\033[0m")
        print(f"  {t_now:>5.1f}s  {true_c:^20}  "
              f"{G}{pred_c:^20}{R}  {conf*100:5.1f}%  "
              f"{snr:4.0f}dB  {G}{'✓' if ok else '✗'}{R}  ({lat:.1f}ms)")

        log.append(dict(t=t_now, true=true_c, pred=pred_c,
                        conf=conf, ok=ok, lat=lat, snr=snr))

    print(f"  {'─'*66}")
    acc = correct / total_pred * 100 if total_pred else 0
    avg_lat = np.mean([r["lat"] for r in log]) if log else 0
    print(f"\n  Toplam: {total_pred}  |  Doğru: {correct}  |  "
          f"Doğruluk: %{acc:.2f}  |  Ort. gecikme: {avg_lat:.2f} ms")

    if log:
        _plot_sim_log(log, classes,
                      os.path.join(RESULTS_DIR, "5_simulasyon_log.png"))
    return log


def _plot_sim_log(log, classes, path):
    ts    = [r["t"]    for r in log]
    ti    = [classes.index(r["true"]) if r["true"] in classes else -1 for r in log]
    pi    = [classes.index(r["pred"]) if r["pred"] in classes else -1 for r in log]
    confs = [r["conf"] for r in log]
    oks   = [r["ok"]   for r in log]
    snrs  = [r["snr"]  for r in log]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    ax.step(ts, ti, where="post", lw=2, color="#1565C0", label="Gerçek")
    ax.scatter([t for t,o in zip(ts,oks) if     o],
               [p for p,o in zip(pi, oks) if     o],
               s=25, color="#43A047", zorder=3, label="Doğru")
    ax.scatter([t for t,o in zip(ts,oks) if not o],
               [p for p,o in zip(pi, oks) if not o],
               s=60, marker="x", color="#E53935", zorder=3, label="Hata")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_ylabel("IHA Sınıfı")
    ax.set_title("Gerçek Zamanlı Simülasyon — Bağımsız Test, Rastgele SNR")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.fill_between(ts, 0, confs, alpha=0.35, color="#7B1FA2")
    ax2.plot(ts, confs, lw=1.2, color="#7B1FA2")
    ax2.axhline(0.5, ls="--", lw=1, color="gray")
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Güven"); ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.fill_between(ts, 0, snrs, alpha=0.3, color="#E65100")
    ax3.plot(ts, snrs, lw=1, color="#E65100")
    ax3.set_ylabel("SNR (dB)"); ax3.set_xlabel("Zaman (s)")
    ax3.set_ylim(0, 25); ax3.grid(alpha=0.3)
    ax3.axhline(5, ls="--", lw=0.8, color="red", label="5 dB eşiği")
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main(skip_generate=False, only_eval=False, sim_seconds=30):
    os.makedirs(MODEL_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  IHA SES SINIFLANDIRMA — DÜRÜST TEST PLATFORMU")
    print(f"{'═'*62}")
    print(f"  Eğitim data : {TRAIN_DIR}  (tohum={SEED_TRAIN})")
    print(f"  Test data   : {TEST_DIR_D}  (tohum={SEED_TEST})")
    print(f"  SNR         : 5–20 dB rastgele (gerçekçi ortam gürültüsü)")
    print(f"  BPF aralığı : 107–233 Hz (birbirine yakın sınıflar)")

    if not skip_generate and not only_eval:
        step_generate()
    else:
        print("\n  Veri üretimi atlandı.")

    if not only_eval:
        step_train()
    else:
        print("\n  Eğitim atlandı.")

    model_path = os.path.join(MODEL_DIR, "iha_model.joblib")
    if not os.path.exists(model_path):
        print(f"\n  HATA: {model_path} bulunamadı."); sys.exit(1)

    print(f"\n  Model yükleniyor...")
    model_pkg = joblib.load(model_path)
    print(f"  Eğitim CV: %{model_pkg['best_accuracy']*100:.2f}  |  "
          f"Yöntem: {model_pkg['best_label']}")

    eval_res   = step_evaluate(model_pkg)
    snr_results= step_snr_stress(model_pkg, snr_levels=(20, 10, 5, 0))
    step_visualize(eval_res, model_pkg, snr_results)
    step_simulate(model_pkg, total_seconds=sim_seconds)

    print(f"\n{'═'*62}")
    print(f"  TAMAMLANDI  →  {RESULTS_DIR}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--only-eval",     action="store_true")
    p.add_argument("--sim-seconds",   type=int, default=30)
    args = p.parse_args()
    main(skip_generate=args.skip_generate,
         only_eval=args.only_eval,
         sim_seconds=args.sim_seconds)
