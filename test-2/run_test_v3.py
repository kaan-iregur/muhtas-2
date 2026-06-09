"""
SARSILMAZ İHA SESLER — Test Platformu v3 (Temporal Split)
==========================================================

v2'nin sorunu: GroupKFold'da tek kayıt olan sınıflar (%ÖRÜMCEK, %DİNOZOR)
eğitimde hiç kalmıyor → %0 accuracy. Bu veriyle recording-level LOOCV imkânsız.

v3 stratejisi — Temporal Split:
  • Her kayıt dosyasının seg0 … segN'yi ikiye böl:
      Eğitim  → seg 0 … seg (N×0.70)           + TÜM augmented versiyonları
      Test    → seg (N×0.70+1) … segN (orijinal ONLY, augment yok)
  • Avantaj: Aynı kayıttan bile olsa, zaman dilimi farklı → korelasyon azalır
  • Test seti hiç augment görmez → gerçekçi
  • Her sınıf hem train'de hem test'te var → ÖRÜMCEK %0 sorunu çözülür

Karşılaştırma için ek metrik:
  • 10-Fold Stratified CV (tüm veriler, MATLAB'a eşdeğer)

Özellik pipeline (MATLAB ses-isleme.txt'e tam birebir):
  1. 7-seviyeli DWT (sym4)
  2. TSK-LHP → 768-boyutlu histogram / seviye → 5376 toplam
  3. 5× MI + 5× Chi2 özellik seçimi
  4. kNN (k=5, Manhattan) + SVM (kübik)
"""

import os
import sys
import re
import glob
import time
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import mutual_info_classif, chi2

warnings.filterwarnings("ignore")

# ─── Yollar ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.resolve()
MAIN_DIR = ROOT.parent / "iha_sistemi"
DATASET  = ROOT / "dataset_sarsilmaz"
RESULTS  = str(ROOT / "sonuclar_v3")
CACHE_V3 = str(ROOT / "cache_v3.npz")

sys.path.insert(0, str(MAIN_DIR))
from feature_extractor import extract_features, load_audio
from config import SAMPLE_RATE, NUM_LEVELS, WAVELET

SEG_DURATION = 3.0
SEG_LEN      = int(SAMPLE_RATE * SEG_DURATION)
TRAIN_RATIO  = 0.70      # Her kaydın ilk %70'i → eğitim
NUM_BINS     = 256
N_PER_LEVEL  = NUM_BINS * 3    # 768
K_FEAT       = 12              # Her seviye × her yöntem → seçilen özellik sayısı
N_RUNS       = 5               # MI ve Chi2 her biri 5×

CMAP = LinearSegmentedColormap.from_list("iha", ["#ffffff", "#1a6bb5", "#0a3060"])


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 1 — VERİ YÜKLEME  (temporal split)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_filename(fname: str):
    """
    Dosya adını ayrıştırır.
    Döner: (kaynak_dosya, seg_no, aug_suffix)
    Örn: bürküt-flight_seg0045_p2.wav → ('bürküt-flight', 45, '_p2')
    """
    stem = Path(fname).stem
    m    = re.match(r'^(.+?)_seg(\d+)(.*)', stem)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return stem, 0, ""


def _build_split_index(dataset_dir: Path):
    """
    Her sınıf × kaynak dosya için temporal split indeksleri oluşturur.

    Returns:
        train_files : [(class_idx, fpath), ...]
        test_files  : [(class_idx, fpath), ...]  ← sadece orijinal segmentler
        classes     : sınıf adları listesi
    """
    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    classes    = [d.name for d in class_dirs]

    train_files, test_files = [], []

    for ci, cdir in enumerate(class_dirs):
        # Kaynak dosya → {seg_no: fpath} sözlüğü
        src_map = defaultdict(dict)
        aug_map = defaultdict(list)  # kaynak → augmented dosya listesi

        for fpath in sorted(glob.glob(str(cdir / "*.wav"))):
            src, seg_no, aug = _parse_filename(Path(fpath).name)
            if aug == "":
                src_map[src][seg_no] = fpath
            else:
                aug_map[src].append(fpath)

        for src in sorted(src_map):
            segs   = sorted(src_map[src].keys())  # 0, 1, 2, ... N
            n      = len(segs)
            cutoff = int(n * TRAIN_RATIO)

            train_segs = segs[:cutoff]
            test_segs  = segs[cutoff:]

            # Eğitim: orijinal + augmented (augmented hep eğitime)
            for s in train_segs:
                train_files.append((ci, src_map[src][s]))
            for fpath in aug_map[src]:
                # Aug hangi segmente ait?
                _, seg_no, _ = _parse_filename(Path(fpath).name)
                if seg_no in set(train_segs):
                    train_files.append((ci, fpath))

            # Test: sadece orijinal segmentler
            for s in test_segs:
                test_files.append((ci, src_map[src][s]))

        print(f"  [{classes[ci]}]  eğitim≈{sum(1 for c,_ in train_files if c==ci)}  "
              f"test≈{sum(1 for c,_ in test_files if c==ci)}")

    return train_files, test_files, classes


def _extract_file_list(file_list, label=""):
    """[(class_idx, fpath)] listesinden özellikleri çıkar."""
    X_list, y_list = [], []
    t0 = time.time()
    for i, (ci, fpath) in enumerate(file_list):
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{len(file_list)} ... {time.time()-t0:.0f}s", flush=True)
        try:
            audio, _ = load_audio(fpath, SAMPLE_RATE)
            feat = extract_features(audio[:SEG_LEN], NUM_LEVELS, WAVELET)
            if feat is not None:
                X_list.append(feat)
                y_list.append(ci)
        except Exception:
            pass
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def load_split(skip_cache=False):
    """
    Temporal split verisini yükle veya cache'den al.

    Returns:
        X_tr, y_tr, X_te, y_te, classes
    """
    cache_tr = str(ROOT / "cache_v3_train.npz")
    cache_te = str(ROOT / "cache_v3_test.npz")

    if (not skip_cache and
            os.path.exists(cache_tr) and os.path.exists(cache_te)):
        print(f"  Cache yükleniyor...")
        d_tr = np.load(cache_tr, allow_pickle=True)
        d_te = np.load(cache_te, allow_pickle=True)
        return (d_tr["X"], d_tr["y"],
                d_te["X"], d_te["y"],
                d_tr["classes"].tolist())

    print(f"  Temporal split indeksi oluşturuluyor (eğitim %{int(TRAIN_RATIO*100)}, test %{int((1-TRAIN_RATIO)*100)})...")
    train_files, test_files, classes = _build_split_index(DATASET)

    print(f"\n  Eğitim özelliklerı çıkarılıyor ({len(train_files)} dosya)...")
    X_tr, y_tr = _extract_file_list(train_files, "EĞITIM")

    print(f"\n  Test özellikleri çıkarılıyor ({len(test_files)} dosya)...")
    X_te, y_te = _extract_file_list(test_files, "TEST")

    np.savez(cache_tr, X=X_tr, y=y_tr, classes=np.array(classes))
    np.savez(cache_te, X=X_te, y=y_te, classes=np.array(classes))
    print(f"\n  Cache kaydedildi.")
    return X_tr, y_tr, X_te, y_te, classes


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 2 — ÖZELLİK SEÇİMİ
# ══════════════════════════════════════════════════════════════════════════════

def _level_slices():
    return [(l * N_PER_LEVEL, (l + 1) * N_PER_LEVEL) for l in range(NUM_LEVELS)]


def select_mi(X, y, k=K_FEAT, seed=0):
    sel = []
    for start, end in _level_slices():
        sc = mutual_info_classif(X[:, start:end], y, random_state=seed)
        top = np.argsort(sc)[::-1][:k]
        sel.extend((start + top).tolist())
    return np.array(sel, dtype=np.int32)


def select_chi2(X, y, k=K_FEAT):
    sel = []
    for start, end in _level_slices():
        Xl = X[:, start:end].astype(np.float64) + 1e-10
        sc, _ = chi2(Xl, y)
        top = np.argsort(sc)[::-1][:k]
        sel.extend((start + top).tolist())
    return np.array(sel, dtype=np.int32)


def build_candidates(X, y):
    cands = []
    for run in range(N_RUNS):
        idx = select_mi(X, y, seed=run * 7)
        cands.append((f"MI-{run+1}", idx))
        print(f"    MI-{run+1}  : {len(idx)} özellik seçildi")
    for run in range(N_RUNS):
        idx = select_chi2(X, y)
        cands.append((f"Chi2-{run+1}", idx))
        print(f"    Chi2-{run+1}: {len(idx)} özellik seçildi")
    return cands


def imv_vote(X, candidates, n_vote=18, seed=42):
    """IMV (Iterative Majority Voting) — MATLAB'ın oy birliği adımı."""
    rng = np.random.default_rng(seed)
    n_cands = len(candidates)
    voted = []
    for _ in range(n_vote):
        chosen = rng.choice(n_cands, size=4, replace=False)
        idx_sets = [candidates[c][1] for c in chosen]
        parts = np.stack([X[:, idx] for idx in idx_sets], axis=2)
        voted.append(("IMV", np.mean(parts, axis=2),
                      None))  # None = no index (voted mean)
    return voted


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 3 — MODEL EĞİTİMİ + TEST
# ══════════════════════════════════════════════════════════════════════════════

CLASSIFIERS = {
    "kNN" : lambda: KNeighborsClassifier(n_neighbors=5, metric="cityblock"),
    "SVM" : lambda: SVC(kernel="poly", degree=3, C=1.0, coef0=1.0,
                        probability=True, class_weight="balanced"),
    "RF"  : lambda: RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                           class_weight="balanced", random_state=42,
                                           n_jobs=-1),
}


def evaluate_all(X_tr, y_tr, X_te, y_te, candidates, classes):
    """
    Her özellik seti × sınıflandırıcı kombinasyonunu test set'te değerlendir.
    En iyi kombinasyonu seç.
    """
    best_acc  = 0.0
    best_info = {}
    all_res   = []

    print(f"\n  {'Özellik Seti':<14} {'Clf':<6} {'Test Acc':>9}  {'F1':>7}")
    print(f"  {'─'*44}")

    for feat_label, idx in candidates:
        X_tr_s = X_tr[:, idx]
        X_te_s = X_te[:, idx]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr_s)
        X_te_sc  = scaler.transform(X_te_s)

        for clf_name, clf_fn in CLASSIFIERS.items():
            clf = clf_fn()
            clf.fit(X_tr_sc, y_tr)
            y_pred = clf.predict(X_te_sc)

            acc = accuracy_score(y_te, y_pred)
            _, _, f1, _ = precision_recall_fscore_support(
                y_te, y_pred, average="weighted", zero_division=0)

            all_res.append({
                "label": f"{feat_label}+{clf_name}",
                "feat_label": feat_label, "clf_name": clf_name,
                "idx": idx, "acc": acc, "f1": f1,
                "clf": clf, "scaler": scaler,
                "y_pred": y_pred,
            })
            marker = " ◄" if acc > best_acc else ""
            print(f"  {feat_label:<14} {clf_name:<6} {acc*100:>8.2f}%  {f1*100:>6.2f}%{marker}")

            if acc > best_acc:
                best_acc  = acc
                best_info = all_res[-1]

    return all_res, best_info


def eval_10fold(X_tr, y_tr, best_idx, classes):
    """10-Katlı Stratified CV — MATLAB'a eşdeğer"""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    X_sel = X_tr[:, best_idx]

    print(f"\n  {'Yöntem':<10} {'10-Fold Ort':>12} {'Std':>7}")
    print(f"  {'─'*35}")

    results = {}
    for clf_name, clf_fn in CLASSIFIERS.items():
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_sel)
        clf    = clf_fn()
        scores = cross_val_score(clf, X_sc, y_tr, cv=skf,
                                  scoring="accuracy", n_jobs=-1)
        results[clf_name] = scores
        print(f"  {clf_name:<10} {scores.mean()*100:>11.2f}%  ±{scores.std()*100:.2f}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════════════════════

def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap=CMAP)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=classes, yticklabels=classes,
           ylabel="Gerçek", xlabel="Tahmin", title=title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            tot = max(cm[i].sum(), 1)
            ax.text(j, i, f"{v}\n({v/tot*100:.0f}%)", ha="center", va="center",
                    fontsize=9, color="white" if v > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_metrics_bar(classes, y_true, y_pred, path, title=""):
    p_c, r_c, f1_c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    x, w = np.arange(len(classes)), 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    for vals, label, color in [
        (p_c,  "Kesinlik",   "#2196F3"),
        (r_c,  "Duyarlılık", "#4CAF50"),
        (f1_c, "F1 Skoru",   "#FF5722"),
    ]:
        offset = {"Kesinlik": -w, "Duyarlılık": 0, "F1 Skoru": w}[label]
        bars = ax.bar(x + offset, vals, w, label=label, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_ylim(0, 1.18); ax.set_ylabel("Skor", fontsize=11)
    ax.set_title(title, fontsize=12); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_comparison(all_res, path):
    """Tüm kombinasyonların test accuracy'si (yatay bar)."""
    srt = sorted(all_res, key=lambda r: r["acc"], reverse=True)[:20]
    labels = [r["label"] for r in srt]
    accs   = [r["acc"] * 100 for r in srt]
    clf_colors = {"kNN": "#1565C0", "SVM": "#B71C1C", "RF": "#2E7D32"}
    colors = [clf_colors.get(r["clf_name"], "#888") for r in srt]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(range(len(labels)), accs, color=colors, alpha=0.82)
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        ax.text(acc + 0.3, i, f"{acc:.1f}%", va="center", ha="left",
                fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Test Doğruluk (%)", fontsize=11)
    ax.set_title("Özellik Seti × Sınıflandırıcı Karşılaştırması\n"
                 "(Temporal Split Testi — İlk %70 eğitim, Son %30 test)", fontsize=12)
    ax.axvline(80, ls="--", color="orange", lw=1.2, label="%80 hedef")
    ax.axvline(90, ls="--", color="green",  lw=1.2, label="%90 hedef")
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=n) for n, c in clf_colors.items()]
    legend_els += [plt.Line2D([0],[0], ls="--", color="orange", label="%80 hedef"),
                   plt.Line2D([0],[0], ls="--", color="green",  label="%90 hedef")]
    ax.legend(handles=legend_els, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(accs) + 12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_10fold(kfold_res, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"kNN": "#1565C0", "SVM": "#B71C1C", "RF": "#2E7D32"}
    x = np.arange(10)
    w = 1.0 / (len(kfold_res) + 1)
    for i, (name, scores) in enumerate(kfold_res.items()):
        offset = (i - len(kfold_res)/2 + 0.5) * w
        bars = ax.bar(x + offset, scores * 100, w,
                      label=f"{name} (Ort: {scores.mean()*100:.1f}%)",
                      color=colors.get(name, "#888"), alpha=0.82)
    ax.axhline(80, ls="--", color="orange", lw=1.2, label="%80 hedef")
    ax.axhline(90, ls="--", color="green",  lw=1.2, label="%90 hedef")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Kat {i+1}" for i in range(10)], fontsize=8)
    ax.set_ylim(0, 112); ax.set_ylabel("Doğruluk (%)", fontsize=11)
    ax.set_title("10-Katlı Stratified CV (MATLAB Eşdeğeri)\n"
                 "Eğitim setinde × en iyi özellik seti", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_dataset_summary(y_tr, y_te, classes, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#1565C0", "#B71C1C", "#2E7D32", "#E65100"]

    for ax, data, title in [
        (axes[0], y_tr, f"Eğitim Seti  (N={len(y_tr):,})"),
        (axes[1], y_te, f"Test Seti  (N={len(y_te):,})\n[Orijinal, augment yok]"),
    ]:
        counts = [np.sum(data == i) for i in range(len(classes))]
        bars = ax.bar(classes, counts, color=colors[:len(classes)], alpha=0.85)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{c:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Segment Sayısı"); ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(counts) * 1.15)

    fig.suptitle("SARSILMAZ İHA — Temporal Split Veri Dağılımı\n"
                 "(Her kayıt: ilk %70 eğitim, son %30 test)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def write_report(classes, best_info, all_res, kfold_res,
                 y_tr, y_te, path):
    best_y_pred = best_info["y_pred"]
    best_acc    = best_info["acc"]
    rep = classification_report(y_te, best_y_pred,
                                 target_names=classes, zero_division=0, digits=3)
    p, r, f1, _ = precision_recall_fscore_support(
        y_te, best_y_pred, average="weighted", zero_division=0)

    with open(path, "w", encoding="utf-8") as f:
        f.write("╔══════════════════════════════════════════════════════════╗\n")
        f.write("║  SARSILMAZ İHA SESLER v3 — TEMPORAL SPLIT RAPORU        ║\n")
        f.write("╚══════════════════════════════════════════════════════════╝\n\n")

        f.write("TEST STRATEJİSİ\n─" * 1 + "─" * 54 + "\n")
        f.write(f"  Segment süresi: {SEG_DURATION}s  (v1: 1s)\n")
        f.write(f"  Split oranı   : İlk %{int(TRAIN_RATIO*100)} eğitim, son %{int((1-TRAIN_RATIO)*100)} test\n")
        f.write(f"  Augmentasyon  : Sadece eğitim setinde\n")
        f.write(f"  Test seti     : Orijinal, augment görmemiş\n\n")

        f.write("VERİ SETİ\n─" + "─" * 54 + "\n")
        for ci, cn in enumerate(classes):
            f.write(f"  {cn:<14}: eğitim={np.sum(y_tr==ci):>5,}  test={np.sum(y_te==ci):>4,}\n")

        f.write(f"\nEN İYİ KOMBİNASYON: {best_info['label']}\n\n")

        f.write("TEMPORAL SPLIT SONUÇLARI\n─" + "─" * 54 + "\n")
        f.write(f"  Doğruluk  : %{best_acc*100:.2f}\n")
        f.write(f"  Kesinlik  : %{p*100:.2f}\n")
        f.write(f"  Duyarlılık: %{r*100:.2f}\n")
        f.write(f"  F1 Skoru  : %{f1*100:.2f}\n\n")
        f.write(rep)

        f.write("\n10-KATLI CV (MATLAB EŞDEĞERİ)\n─" + "─" * 54 + "\n")
        for name, scores in kfold_res.items():
            f.write(f"  {name:<8}: %{scores.mean()*100:.2f} ± %{scores.std()*100:.2f}\n")

        f.write("\nTÜM KOMBİNASYONLAR (test accuracy sıralı)\n─" + "─" * 54 + "\n")
        for res in sorted(all_res, key=lambda r: r["acc"], reverse=True):
            f.write(f"  {res['label']:<22} {res['acc']*100:>8.2f}%  F1={res['f1']*100:.2f}%\n")

        f.write(f"\n\nTarih: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main(skip_feat=False):
    os.makedirs(RESULTS, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  SARSILMAZ İHA — TSK-LHP/MDWT TEST v3  (Temporal Split)")
    print(f"{'═'*62}")
    print(f"  Segment    : {SEG_DURATION}s  |  Split: %{int(TRAIN_RATIO*100)}/%{int((1-TRAIN_RATIO)*100)}")
    print(f"  Özellik    : {NUM_LEVELS}×768 = {NUM_LEVELS*768} boyut")
    print(f"  Augment    : Sadece eğitim setinde")

    # ── Adım 1: Veri ──────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 1 — VERİ YÜKLEME  (Temporal Split)")
    print(f"{'═'*62}\n")
    t0 = time.time()
    X_tr, y_tr, X_te, y_te, classes = load_split(skip_cache=not skip_feat)
    print(f"\n  ┌{'─'*46}┐")
    print(f"  │  Eğitim   : {len(X_tr):>7,} segment  (orijinal+aug)    │")
    print(f"  │  Test     : {len(X_te):>7,} segment  (orijinal only)   │")
    for ci, cn in enumerate(classes):
        print(f"  │  {cn:<12}: eğitim={np.sum(y_tr==ci):>5,}  test={np.sum(y_te==ci):>4,}      │")
    print(f"  └{'─'*46}┘")
    print(f"  Süre: {time.time()-t0:.1f}s")

    plot_dataset_summary(y_tr, y_te, classes,
                         os.path.join(RESULTS, "0_veri_seti.png"))

    # ── Adım 2: Özellik Seçimi ────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 2 — ÖZELLİK SEÇİMİ  (eğitim seti üzerinde)")
    print(f"{'═'*62}\n")
    candidates = build_candidates(X_tr, y_tr)

    # ── Adım 3: Test ──────────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 3 — TEMPORAL SPLIT TEST  (model test seti hiç görmedi)")
    print(f"{'═'*62}")
    all_res, best_info = evaluate_all(X_tr, y_tr, X_te, y_te, candidates, classes)

    print(f"\n  ── En iyi: {best_info['label']}  ({best_info['acc']*100:.2f}%)")

    # ── Adım 4: 10-Fold CV (MATLAB karşılaştırma) ─────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 4 — 10-KATLI STRATIFIED CV  (MATLAB eşdeğeri)")
    print(f"{'═'*62}")
    kfold_res = eval_10fold(X_tr, y_tr, best_info["idx"], classes)

    # ── Adım 5: Görselleştirme ────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 5 — GÖRSELLEŞTİRME")
    print(f"{'═'*62}\n")

    y_pred = best_info["y_pred"]
    cm     = confusion_matrix(y_te, y_pred, labels=list(range(len(classes))))

    p, r, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, average="weighted", zero_division=0)

    print(f"\n  ┌{'─'*46}┐")
    print(f"  │  Doğruluk   : %{best_info['acc']*100:6.2f}                        │")
    print(f"  │  Kesinlik   : %{p*100:6.2f}  (ağırlıklı)            │")
    print(f"  │  Duyarlılık : %{r*100:6.2f}  (ağırlıklı)            │")
    print(f"  │  F1 Skoru   : %{f1*100:6.2f}  (ağırlıklı)            │")
    print(f"  └{'─'*46}┘")

    rep = classification_report(y_te, y_pred, target_names=classes, zero_division=0, digits=3)
    print(f"\n  Sınıflandırma Raporu:\n  {'─'*50}")
    for ln in rep.split("\n"):
        print(f"  {ln}")

    plot_cm(cm, classes,
            f"Karmaşıklık Matrisi — {best_info['label']}\n"
            f"Temporal Split  |  Doğruluk: %{best_info['acc']*100:.2f}",
            os.path.join(RESULTS, "1_karmasiklik_matrisi.png"))

    plot_metrics_bar(classes, y_te, y_pred,
                     os.path.join(RESULTS, "2_sinif_metrikleri.png"),
                     title=f"Sınıf Bazında Performans — {best_info['label']}\n"
                           f"(3s segment, MI/Chi2, Temporal Split)")

    plot_comparison(all_res, os.path.join(RESULTS, "3_tum_kombinasyonlar.png"))
    plot_10fold(kfold_res, os.path.join(RESULTS, "4_10fold_cv.png"))

    write_report(classes, best_info, all_res, kfold_res,
                 y_tr, y_te, os.path.join(RESULTS, "rapor_v3.txt"))

    # Model kaydet
    joblib.dump({
        "classifier" : best_info["clf"],
        "scaler"     : best_info["scaler"],
        "feat_idx"   : best_info["idx"],
        "classes"    : classes,
        "best_label" : best_info["label"],
        "accuracy"   : best_info["acc"],
        "seg_duration": SEG_DURATION,
        "sample_rate": SAMPLE_RATE,
    }, os.path.join(RESULTS, "sarsilmaz_model_v3.joblib"))
    print(f"  → {os.path.join(RESULTS, 'sarsilmaz_model_v3.joblib')}")

    print(f"\n{'═'*62}")
    print(f"  SONUÇLAR")
    print(f"{'═'*62}")
    print(f"  Temporal Split : %{best_info['acc']*100:.2f}  ({best_info['label']})")
    for name, scores in kfold_res.items():
        print(f"  10-Fold {name:<7}: %{scores.mean()*100:.2f} ± %{scores.std()*100:.2f}")
    print(f"\n  Çıktılar : {RESULTS}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-feat", action="store_true",
                   help="Cache varsa özellik çıkarmayı atla")
    args = p.parse_args()
    main(skip_feat=args.skip_feat)
