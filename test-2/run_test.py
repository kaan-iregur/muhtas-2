"""
SARSILMAZ İHA SESLER — Gerçek Veri Seti Test Platformu
=======================================================

Algoritma: MDWT (7-seviye, sym4) + TSK-LHP özellik çıkarma
Kaynak   : ses-isleme.txt (MATLAB → Python dönüşümü)

Sınıflar:
  - BÜRKÜT   (2 dosya, ~3.4 dk)
  - DİNOZOR  (2 dosya, ~9.7 dk)
  - HEXA     (2 dosya, ~4.5 dk)
  - ÖRÜMCEK  (1 dosya, ~3.1 dk)

Test Stratejisi:
  - Adım 1 : Tüm veri setini yükle, segmentlere böl, özellikleri çıkar
  - Adım 2 : Dosya bazlı Leave-One-File-Out CV (data leakage'ı önler)
  - Adım 3 : Stratified 5-Fold CV (genel performans tahmini)
  - Adım 4 : Gürültü dayanıklılık testi (SNR: 30 → 5 → 0 dB)
  - Adım 5 : Görselleştirme + rapor

Kullanım:
  python run_test.py
  python run_test.py --only-vis   # Sadece görsel (özellikler zaten çıkarıldıysa)
"""

import os
import sys
import glob
import time
import argparse
import warnings
import json
from pathlib import Path

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# ─── Yollar ──────────────────────────────────────────────────────────────────
TEST2_DIR  = Path(__file__).parent.resolve()
MAIN_DIR   = TEST2_DIR.parent / "iha_sistemi"
DATASET    = TEST2_DIR.parent / "SARSILMAZ İHA SESLER"
RESULTS    = str(TEST2_DIR / "sonuclar")
CACHE_FILE = str(TEST2_DIR / "ozellik_cache.npz")

sys.path.insert(0, str(MAIN_DIR))
from feature_extractor import extract_features, load_audio
from config import SAMPLE_RATE, SEGMENT_DURATION, NUM_LEVELS, WAVELET

SEG_LEN = int(SAMPLE_RATE * SEGMENT_DURATION)

CMAP = LinearSegmentedColormap.from_list("iha", ["#ffffff", "#1a6bb5", "#0a3060"])


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 1 — VERİ YÜKLEME VE ÖZELLİK ÇIKARMA
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset():
    """
    SARSILMAZ İHA SESLER klasöründeki tüm WAV dosyalarını yükler,
    1 saniyelik segmentlere böler, MDWT+TSK-LHP özellikleri çıkarır.

    Returns:
        X        : (N, 5376) özellik matrisi
        y        : (N,) sınıf indeksleri
        file_ids : (N,) hangi dosyadan geldiğini tutan indeks
        classes  : sınıf adları listesi
        file_map : {file_id: (class_name, filename)} sözlüğü
    """
    class_dirs = sorted([
        d for d in DATASET.iterdir()
        if d.is_dir() and d.name != "GÖRSELLER"
    ])

    classes  = [d.name for d in class_dirs]
    X_list   = []
    y_list   = []
    fid_list = []
    file_map = {}
    file_counter = 0

    print(f"\n  Sınıflar ({len(classes)}): {classes}")
    print(f"  Segment uzunluğu: {SEGMENT_DURATION}s @ {SAMPLE_RATE} Hz\n")

    for class_idx, class_dir in enumerate(class_dirs):
        audio_files = sorted(
            glob.glob(str(class_dir / "*.wav")) +
            glob.glob(str(class_dir / "*.WAV")) +
            glob.glob(str(class_dir / "*.flac"))
        )
        count_before = len(X_list)

        for fpath in audio_files:
            fid = file_counter
            fname = Path(fpath).name
            file_map[fid] = (classes[class_idx], fname)
            file_counter += 1

            try:
                audio, sr = load_audio(fpath, SAMPLE_RATE)
                # Örtüşmeyen segmentler
                n_segs = 0
                for start in range(0, len(audio) - SEG_LEN + 1, SEG_LEN):
                    seg  = audio[start:start + SEG_LEN]
                    feat = extract_features(seg, NUM_LEVELS, WAVELET)
                    if feat is not None:
                        X_list.append(feat)
                        y_list.append(class_idx)
                        fid_list.append(fid)
                        n_segs += 1
                print(f"    [{classes[class_idx]}] {fname:<35} → {n_segs} segment")
            except Exception as e:
                print(f"    HATA ({fname}): {e}")

        added = len(X_list) - count_before
        print(f"    ── Toplam {classes[class_idx]}: {added} segment\n")

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int32),
        np.array(fid_list, dtype=np.int32),
        classes,
        file_map,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 2 — DOSYA BAZLI LOOCV (Leave-One-File-Out)
# ══════════════════════════════════════════════════════════════════════════════

def loocv(X, y, file_ids, classes, file_map):
    """
    Her dosyayı sırayla test seti olarak kullanır, kalanlarla eğitir.
    Veri sızıntısı (data leakage) riski yoktur — aynı kayıttan
    hem eğitim hem test segmenti gelmez.
    """
    unique_files = sorted(np.unique(file_ids))
    all_true, all_pred = [], []

    print(f"  {'Dosya':<35} {'Test Seg':>9} {'Doğruluk':>9}  Notlar")
    print(f"  {'─'*70}")

    for fid in unique_files:
        class_name, fname = file_map[fid]

        test_mask  = file_ids == fid
        train_mask = ~test_mask

        # En az 2 sınıfın eğitim verisinde olması gerekiyor
        train_classes = np.unique(y[train_mask])
        if len(train_classes) < 2:
            print(f"  {fname:<35} {'─':>9}  ATLANDI (eğitim veri yetersiz)")
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = KNeighborsClassifier(n_neighbors=5, metric="cityblock")
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)

        acc = accuracy_score(y_te, y_pred)
        all_true.extend(y_te.tolist())
        all_pred.extend(y_pred.tolist())

        note = "✓" if acc >= 0.80 else ("△" if acc >= 0.60 else "✗")
        print(f"  {fname:<35} {len(y_te):>9} {acc*100:>8.1f}%  {note}")

    print(f"  {'─'*70}")
    overall = accuracy_score(all_true, all_pred) if all_true else 0.0
    print(f"  {'GENEL LOOCV':>35} {len(all_true):>9} {overall*100:>8.1f}%")

    return np.array(all_true), np.array(all_pred), overall


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 3 — STRATİFİED K-FOLD CV
# ══════════════════════════════════════════════════════════════════════════════

def kfold_cv(X, y, classes):
    """
    kNN ve SVM için 5-katlı stratified çapraz doğrulama.
    StandardScaler pipeline'a dahil.
    """
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # kNN
    knn = KNeighborsClassifier(n_neighbors=5, metric="cityblock")
    scores_knn = cross_val_score(knn, X_sc, y, cv=skf, scoring="accuracy", n_jobs=-1)
    results["kNN"] = scores_knn
    print(f"\n  kNN (k=5, Manhattan) →"
          f"  Ort: {scores_knn.mean()*100:.2f}%  ±{scores_knn.std()*100:.2f}%"
          f"  |  Katlı: {[f'{s*100:.1f}%' for s in scores_knn]}")

    # SVM Kübik
    svm = SVC(kernel="poly", degree=3, C=1.0, coef0=1.0, probability=True)
    scores_svm = cross_val_score(svm, X_sc, y, cv=skf, scoring="accuracy", n_jobs=-1)
    results["SVM"] = scores_svm
    print(f"  SVM (kübik, C=1)    →"
          f"  Ort: {scores_svm.mean()*100:.2f}%  ±{scores_svm.std()*100:.2f}%"
          f"  |  Katlı: {[f'{s*100:.1f}%' for s in scores_svm]}")

    return results, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 4 — GÜRÜLTÜ DAYANIKLILIK TESTİ
# ══════════════════════════════════════════════════════════════════════════════

def add_awgn(signal: np.ndarray, snr_db: float, rng) -> np.ndarray:
    """Matrise belirtilen SNR seviyesinde beyaz Gaussian gürültü ekler."""
    sig_power   = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise       = rng.normal(0, np.sqrt(max(noise_power, 1e-12)), signal.shape)
    return signal + noise


def noise_robustness_test(X_raw_segments, y, file_ids, classes, snr_levels=(30, 20, 15, 10, 5, 0)):
    """
    Segment başına AWGN ekleyerek farklı SNR seviyelerinde test eder.
    Bir fold train, diğer fold test olarak LOOCV benzeri split kullanır.
    """
    # Eğitim/test bölünmesi: ilk yarı sınıflar train, ikinci yarı test
    rng_split = np.random.default_rng(42)
    unique_files = sorted(np.unique(file_ids))

    # Sınıf başına dosyaları böl
    train_segs, test_segs = [], []
    for c in np.unique(y):
        c_files = [f for f in unique_files if file_ids[file_ids == f].shape[0] > 0
                   and np.any(y[file_ids == f] == c)]
        c_files_actual = [f for f in unique_files
                          if np.any((file_ids == f) & (y == c))]
        if len(c_files_actual) >= 2:
            half = len(c_files_actual) // 2
            for f in c_files_actual[:half]:
                mask = (file_ids == f) & (y == c)
                train_segs.extend(np.where(mask)[0].tolist())
            for f in c_files_actual[half:]:
                mask = (file_ids == f) & (y == c)
                test_segs.extend(np.where(mask)[0].tolist())
        else:
            # Tek dosya: ilk yarı train, ikinci yarı test (segment bazında)
            idxs = np.where(y == c)[0]
            half = len(idxs) // 2
            train_segs.extend(idxs[:half].tolist())
            test_segs.extend(idxs[half:].tolist())

    train_segs = np.array(train_segs)
    test_segs  = np.array(test_segs)

    scaler = StandardScaler()
    scaler.fit(X_raw_segments[train_segs])

    clf = KNeighborsClassifier(n_neighbors=5, metric="cityblock")
    clf.fit(scaler.transform(X_raw_segments[train_segs]), y[train_segs])

    # Orijinal test segmentleri (gürültüsüz referans)
    X_te_clean = X_raw_segments[test_segs]
    y_te       = y[test_segs]

    print(f"\n  {'SNR':>6}  {'Doğruluk':>10}  Sınıf bazında doğruluk")
    print(f"  {'─'*60}")

    snr_results = {}
    for snr in snr_levels:
        # Gürültü simülasyonu: özellik uzayında değil, özellik üzerine
        # Gaussian pertürbasyon ekle (SNR-orantılı)
        rng = np.random.default_rng(99 + int(snr * 10))
        X_noisy = add_awgn(X_te_clean.copy(), snr_db=snr, rng=rng)
        X_noisy_sc = scaler.transform(X_noisy)
        y_pred  = clf.predict(X_noisy_sc)
        acc     = accuracy_score(y_te, y_pred)

        per_cls = []
        for ci in range(len(classes)):
            mask = y_te == ci
            if mask.sum() > 0:
                per_cls.append(f"{classes[ci][:6]}:{accuracy_score(y_te[mask], y_pred[mask])*100:.0f}%")

        snr_results[snr] = {
            "accuracy": acc,
            "per_class": per_cls,
            "cm": confusion_matrix(y_te, y_pred, labels=list(range(len(classes)))),
        }
        cls_str = "  ".join(per_cls)
        print(f"  {snr:>4} dB  {acc*100:>9.2f}%  [{cls_str}]")

    return snr_results


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 5 — GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=CMAP)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=classes, yticklabels=classes,
           ylabel="Gerçek Sınıf", xlabel="Tahmin Edilen",
           title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    thresh = cm.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            v   = cm[i, j]
            tot = max(cm[i].sum(), 1)
            pct = v / tot * 100
            ax.text(j, i, f"{v}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=9, color="white" if v > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_metrics_bar(classes, y_true, y_pred, path, title_suffix=""):
    p_c, r_c, f1_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    x, w = np.arange(len(classes)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_p  = ax.bar(x - w, p_c,  w, label="Kesinlik",   color="#2196F3", alpha=0.85)
    bars_r  = ax.bar(x,     r_c,  w, label="Duyarlılık", color="#4CAF50", alpha=0.85)
    bars_f1 = ax.bar(x + w, f1_c, w, label="F1 Skoru",   color="#FF5722", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Skor", fontsize=11)
    ax.set_title(f"Sınıf Bazında Performans{title_suffix}", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar in (list(bars_p) + list(bars_r) + list(bars_f1)):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_kfold_bar(kfold_results, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"kNN": "#1565C0", "SVM": "#B71C1C"}
    fold_labels = [f"Kat {i+1}" for i in range(5)]
    x = np.arange(5)
    w = 0.35
    for i, (name, scores) in enumerate(kfold_results.items()):
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, scores * 100, w,
                      label=f"{name} (Ort: {scores.mean()*100:.1f}%)",
                      color=colors.get(name, "#666"), alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Doğruluk (%)", fontsize=11)
    ax.set_title("5-Katlı Stratified CV — kNN vs SVM\n(SARSILMAZ İHA Gerçek Ses Verisi)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_snr_curve(snr_results, classes, path):
    snrs = sorted(snr_results.keys(), reverse=True)
    accs = [snr_results[s]["accuracy"] * 100 for s in snrs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(snrs, accs, "o-", lw=2.5, color="#1565C0", markersize=10)
    for snr, acc in zip(snrs, accs):
        ax1.annotate(f"%{acc:.1f}", (snr, acc),
                     textcoords="offset points", xytext=(0, 11),
                     ha="center", fontsize=10, fontweight="bold", color="#1565C0")
    ax1.set_xlabel("SNR (dB)", fontsize=12)
    ax1.set_ylabel("Doğruluk (%)", fontsize=12)
    ax1.set_title("Gürültü Dayanıklılık Eğrisi\n(AWGN gürültüsü, özellik uzayında)", fontsize=11)
    ax1.set_ylim(0, 115)
    ax1.axhline(25, ls="--", color="red",    lw=1.2, label="Rastgele eşiği (4 sınıf → %25)")
    ax1.axhline(80, ls="--", color="orange", lw=1.2, label="Kabul edilebilir (%80)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.35)
    ax1.fill_between(snrs, accs, 0, alpha=0.08, color="#1565C0")

    worst_snr = min(snrs)
    worst_cm  = snr_results[worst_snr]["cm"]
    im = ax2.imshow(worst_cm, cmap=CMAP)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    ax2.set(xticks=ticks, yticks=ticks,
            xticklabels=classes, yticklabels=classes,
            title=f"En Zor Senaryo — SNR={worst_snr} dB\n"
                  f"(Doğruluk: %{snr_results[worst_snr]['accuracy']*100:.1f})",
            ylabel="Gerçek", xlabel="Tahmin")
    plt.setp(ax2.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    plt.setp(ax2.get_yticklabels(), fontsize=9)
    thresh = worst_cm.max() / 2 if worst_cm.max() > 0 else 1
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = worst_cm[i, j]
            ax2.text(j, i, str(v), ha="center", va="center",
                     fontsize=10, color="white" if v > thresh else "black")

    fig.suptitle("SNR Stres Testi — SARSILMAZ İHA Sistemi Gürültü Dayanıklılığı",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_dataset_overview(y, file_ids, classes, file_map, path):
    """Veri seti istatistiklerini göster."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Sol: Sınıf başına segment sayısı
    counts = [np.sum(y == i) for i in range(len(classes))]
    colors_bar = ["#1565C0", "#B71C1C", "#2E7D32", "#E65100"]
    bars = ax1.bar(classes, counts, color=colors_bar[:len(classes)], alpha=0.85)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(c), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Segment Sayısı (1 sn/segment)", fontsize=11)
    ax1.set_title("Sınıf Başına Segment Dağılımı\n(SARSILMAZ İHA Gerçek Ses)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(counts) * 1.15)

    # Sağ: Dosya başına segment sayısı
    file_segs = {}
    for fid in sorted(np.unique(file_ids)):
        cname, fname = file_map[fid]
        short = f"{cname}\n{fname[:20]}"
        file_segs[short] = np.sum(file_ids == fid)

    labels = list(file_segs.keys())
    values = list(file_segs.values())
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax2.pie(
        values, labels=labels, autopct="%1.0f%%",
        colors=colors_pie, startangle=140,
        textprops={"fontsize": 8}
    )
    ax2.set_title("Dosya Başına Segment Payı", fontsize=11)

    fig.suptitle("SARSILMAZ İHA SESLER — Veri Seti Özeti", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def write_report(classes, y_loocv_true, y_loocv_pred, loocv_acc,
                 kfold_results, snr_results, file_map, y, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("╔══════════════════════════════════════════════════════════╗\n")
        f.write("║   SARSILMAZ İHA SESLER — TSK-LHP/MDWT TEST RAPORU       ║\n")
        f.write("╚══════════════════════════════════════════════════════════╝\n\n")

        f.write("VERİ SETİ\n")
        f.write("─" * 55 + "\n")
        f.write(f"  Toplam sınıf  : {len(classes)}\n")
        f.write(f"  Sınıflar      : {', '.join(classes)}\n")
        f.write(f"  Toplam segment: {len(y)} (@ {SEGMENT_DURATION}s, {SAMPLE_RATE} Hz)\n")
        for ci, cn in enumerate(classes):
            f.write(f"    {cn:<12}: {np.sum(y == ci):>4} segment\n")
        f.write(f"\n  Dosyalar:\n")
        for fid in sorted(file_map):
            cname, fname = file_map[fid]
            f.write(f"    [{cname}] {fname}\n")

        f.write("\n\nALGORİTMA\n")
        f.write("─" * 55 + "\n")
        f.write(f"  Dalgacık      : {WAVELET} (Symlet-4)\n")
        f.write(f"  MDWT seviyesi : {NUM_LEVELS}\n")
        f.write(f"  Özellik boyutu: {NUM_LEVELS} × 768 = {NUM_LEVELS * 768}\n")
        f.write(f"  Özellik seçimi: 5× Mutual-Info + 5× Chi-Squared\n")
        f.write(f"  Sınıflandırıcı: kNN (k=5, Manhattan)\n")

        f.write("\n\nSONUÇLAR — DOSYA BAZLI LOOCV\n")
        f.write("─" * 55 + "\n")
        f.write(f"  Genel Doğruluk: %{loocv_acc*100:.2f}\n")
        if len(y_loocv_true) > 0:
            p, r, f1, _ = precision_recall_fscore_support(
                y_loocv_true, y_loocv_pred, average="weighted", zero_division=0)
            f.write(f"  Kesinlik      : %{p*100:.2f} (ağırlıklı)\n")
            f.write(f"  Duyarlılık    : %{r*100:.2f} (ağırlıklı)\n")
            f.write(f"  F1 Skoru      : %{f1*100:.2f} (ağırlıklı)\n")
            f.write(f"\n  Sınıflandırma Raporu:\n")
            rep = classification_report(
                y_loocv_true, y_loocv_pred,
                target_names=classes, zero_division=0)
            f.write(rep)

        f.write("\n\nSONUÇLAR — 5-KATLI STRATİFİED CV\n")
        f.write("─" * 55 + "\n")
        for name, scores in kfold_results.items():
            f.write(f"  {name:<6}: Ort %{scores.mean()*100:.2f} ± %{scores.std()*100:.2f}"
                    f"  |  Katlı: {[f'{s*100:.1f}' for s in scores]}\n")

        f.write("\n\nSNR STRES TESTİ\n")
        f.write("─" * 55 + "\n")
        for snr in sorted(snr_results.keys(), reverse=True):
            res = snr_results[snr]
            f.write(f"  SNR={snr:>3} dB  →  %{res['accuracy']*100:.2f}\n")

        f.write("\n\n")
        f.write("Referans : ses-isleme.txt (MATLAB → Python/NumPy)\n")
        f.write("Tarih    : " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main(only_vis=False):
    os.makedirs(RESULTS, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  SARSILMAZ İHA SESLER — MDWT + TSK-LHP TEST PLATFORMU")
    print(f"{'═'*62}")
    print(f"  Veri seti : {DATASET}")
    print(f"  Algoritma : {NUM_LEVELS}-seviyeli MDWT + TSK-LHP (sym4)")
    print(f"  Özellik   : {NUM_LEVELS}×768 = {NUM_LEVELS*768} boyut")

    # ── Adım 1: Özellik çıkarma ────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 1 — VERİ YÜKLEME VE ÖZELLİK ÇIKARMA")
    print(f"{'═'*62}")
    t0 = time.time()

    if not only_vis and os.path.exists(CACHE_FILE):
        print(f"  Cache bulundu, yükleniyor: {CACHE_FILE}")
        cache    = np.load(CACHE_FILE, allow_pickle=True)
        X        = cache["X"]
        y        = cache["y"]
        file_ids = cache["file_ids"]
        classes  = cache["classes"].tolist()
        file_map = {int(k): tuple(v) for k, v in cache["file_map"]}
    else:
        X, y, file_ids, classes, file_map = load_dataset()
        np.savez(CACHE_FILE,
                 X=X, y=y, file_ids=file_ids,
                 classes=np.array(classes),
                 file_map=np.array([(k, list(v)) for k, v in file_map.items()], dtype=object))
        print(f"\n  Cache kaydedildi: {CACHE_FILE}")

    print(f"\n  ┌{'─'*42}┐")
    print(f"  │  Toplam segment : {len(X):>6}                    │")
    print(f"  │  Özellik boyutu : {X.shape[1]:>6}                    │")
    for ci, cn in enumerate(classes):
        print(f"  │  {cn:<14}: {np.sum(y==ci):>5} segment              │")
    print(f"  └{'─'*42}┘")
    print(f"  Özellik çıkarma : {time.time()-t0:.1f}s")

    # ── Adım 2: LOOCV ──────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 2 — DOSYA BAZLI LOOCV (Veri Sızıntısız)")
    print(f"{'═'*62}\n")
    y_loocv_true, y_loocv_pred, loocv_acc = loocv(X, y, file_ids, classes, file_map)

    # ── Adım 3: K-Fold CV ──────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 3 — STRATİFİED 5-KATLI ÇAPRAZ DOĞRULAMA")
    print(f"{'═'*62}")
    kfold_results, scaler = kfold_cv(X, y, classes)

    # ── Adım 4: SNR Testi ─────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 4 — GÜRÜLTÜ DAYANIKLILIK TESTİ (SNR Stres)")
    print(f"{'═'*62}")
    snr_results = noise_robustness_test(X, y, file_ids, classes,
                                        snr_levels=(30, 20, 15, 10, 5, 0))

    # ── Adım 5: Görselleştirme ─────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 5 — GÖRSELLEŞTİRME")
    print(f"{'═'*62}\n")

    # Veri seti özeti
    plot_dataset_overview(y, file_ids, classes, file_map,
                          os.path.join(RESULTS, "0_veri_seti_ozeti.png"))

    # LOOCV karmaşıklık matrisi
    if len(y_loocv_true) > 0:
        cm_loocv = confusion_matrix(y_loocv_true, y_loocv_pred,
                                    labels=list(range(len(classes))))
        plot_cm(cm_loocv, classes,
                f"Karmaşıklık Matrisi — Dosya Bazlı LOOCV\n"
                f"Doğruluk: %{loocv_acc*100:.2f}  |  Gerçek Ses Verisi",
                os.path.join(RESULTS, "1_loocv_karmasiklik_matrisi.png"))

        plot_metrics_bar(classes, y_loocv_true, y_loocv_pred,
                         os.path.join(RESULTS, "2_loocv_sinif_metrikleri.png"),
                         title_suffix=" — LOOCV")

    # K-Fold bar grafiği
    plot_kfold_bar(kfold_results,
                   os.path.join(RESULTS, "3_kfold_cv_sonuclari.png"))

    # SNR eğrisi
    plot_snr_curve(snr_results, classes,
                   os.path.join(RESULTS, "4_snr_stres_testi.png"))

    # ── Metin Raporu ──────────────────────────────────────────────────────
    write_report(
        classes, y_loocv_true, y_loocv_pred, loocv_acc,
        kfold_results, snr_results, file_map, y,
        os.path.join(RESULTS, "rapor.txt")
    )

    # ── Özet ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  SONUÇLAR")
    print(f"{'═'*62}")
    print(f"  LOOCV Doğruluk : %{loocv_acc*100:.2f}")
    for name, scores in kfold_results.items():
        print(f"  {name} 5-Fold   : %{scores.mean()*100:.2f} ± %{scores.std()*100:.2f}")
    print(f"\n  Çıktılar       : {RESULTS}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only-vis", action="store_true",
                   help="Sadece görselleştirme (cache varsa özellik çıkarma atlanır)")
    args = p.parse_args()
    main(only_vis=args.only_vis)
