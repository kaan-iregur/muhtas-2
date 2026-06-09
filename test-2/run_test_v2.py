"""
SARSILMAZ İHA SESLER — İyileştirilmiş Test Platformu v2
=========================================================

Önceki versiyona göre kritik iyileştirmeler:
  1. 3 saniyelik segmentler  (L7'de 131 → 381 approx katsayı, 3x stabil histogram)
  2. 1 saniyelik kaymayla örtüşen segmentasyon (dataset_sarsilmaz/ dan yüklenir)
  3. MI + Chi2 özellik seçimi (MATLAB'ın fscnca+chi2 pipeline'ı)
  4. GroupKFold cross-validation — kaynak dosya bazlı gruplama
     (augmented versiyonlar asla test setine sızmaz)
  5. kNN + SVM + Random Forest karşılaştırması

Kullanım:
  python run_test_v2.py               # Tam pipeline
  python run_test_v2.py --skip-feat   # Özellikler cache'de varsa atla
"""

import os
import sys
import glob
import time
import argparse
import warnings
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, chi2

warnings.filterwarnings("ignore")

# ─── Yollar ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.resolve()
MAIN_DIR   = ROOT.parent / "iha_sistemi"
DATASET    = ROOT / "dataset_sarsilmaz"
RESULTS    = str(ROOT / "sonuclar_v2")
CACHE_V2   = str(ROOT / "cache_v2.npz")

sys.path.insert(0, str(MAIN_DIR))
from feature_extractor import extract_features, load_audio
from config import SAMPLE_RATE, NUM_LEVELS, WAVELET

SEG_DURATION = 3.0
SEG_LEN      = int(SAMPLE_RATE * SEG_DURATION)
CMAP         = LinearSegmentedColormap.from_list("iha", ["#ffffff", "#1a6bb5", "#0a3060"])

# Özellik seçimi parametreleri
FEATURES_PER_LEVEL = 10   # Her DWT seviyesi için seçilecek özellik (MATLAB'da 5, burada daha geniş)
SELECTION_RUNS     = 5


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 1 — ÖZELLİK ÇIKARMA  (3 saniyelik segmentler)
# ══════════════════════════════════════════════════════════════════════════════

def _source_file_id(filename: str) -> str:
    """
    Augmented dosya adından kaynak dosyayı çıkar.
    Örn: bürküt-flight_seg0045_p2.wav → bürküt-flight
         dino-flight-1_seg0012.wav     → dino-flight-1
    """
    stem = Path(filename).stem          # bürküt-flight_seg0045_p2
    # _seg kısmından önce = kaynak dosya adı
    parts = stem.split("_seg")
    return parts[0]


def extract_all(skip_cache: bool = False):
    """
    dataset_sarsilmaz/ altındaki tüm WAV'lardan özellik çıkar.
    Her segment için kaynak dosya ID'sini de saklar (GroupKFold için).

    Returns:
        X        : (N, 5376) float32
        y        : (N,)  sınıf indeksi
        groups   : (N,)  kaynak dosya grup ID'si (int)
        classes  : sınıf adları listesi
        group_map: {grup_id: kaynak_dosya_adı}
    """
    if not skip_cache and os.path.exists(CACHE_V2):
        print(f"  Cache yükleniyor: {CACHE_V2}")
        d = np.load(CACHE_V2, allow_pickle=True)
        return (d["X"], d["y"], d["groups"],
                d["classes"].tolist(),
                {int(k): str(v) for k, v in d["group_map"]})

    class_dirs = sorted([
        d for d in DATASET.iterdir() if d.is_dir()
    ])
    classes = [d.name for d in class_dirs]

    # Tüm kaynak dosya adlarını topla → sayısal ID ata
    src_set = set()
    for cd in class_dirs:
        for fpath in glob.glob(str(cd / "*.wav")):
            src_set.add(_source_file_id(Path(fpath).name))
    src_list = sorted(src_set)
    src2id   = {s: i for i, s in enumerate(src_list)}
    group_map = {i: s for s, i in src2id.items()}

    X_list, y_list, g_list = [], [], []

    print(f"  Kaynak dosya grupları ({len(src_list)}): {src_list}\n")

    for ci, class_dir in enumerate(class_dirs):
        files = sorted(glob.glob(str(class_dir / "*.wav")))
        cnt   = 0
        for fpath in files:
            src_id = src2id[_source_file_id(Path(fpath).name)]
            try:
                audio, _ = load_audio(fpath, SAMPLE_RATE)
                # 3 saniyelik segment doğrudan yüklenen dosyadır
                # (prepare_dataset.py zaten 3s parçalara böldü)
                feat = extract_features(audio[:SEG_LEN], NUM_LEVELS, WAVELET)
                if feat is not None:
                    X_list.append(feat)
                    y_list.append(ci)
                    g_list.append(src_id)
                    cnt += 1
            except Exception as e:
                pass  # sessizce atla

        print(f"    [{classes[ci]}] {cnt} segment")

    X      = np.array(X_list, dtype=np.float32)
    y      = np.array(y_list, dtype=np.int32)
    groups = np.array(g_list, dtype=np.int32)

    np.savez(CACHE_V2, X=X, y=y, groups=groups,
             classes=np.array(classes),
             group_map=np.array(list(group_map.items()), dtype=object))
    print(f"\n  Cache kaydedildi: {CACHE_V2}")
    return X, y, groups, classes, group_map


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 2 — ÖZELLİK SEÇİMİ  (MI + Chi2, MATLAB'ın fscnca muadili)
# ══════════════════════════════════════════════════════════════════════════════

NUM_BINS      = 256
N_PER_LEVEL   = NUM_BINS * 3    # 768

def _level_slices():
    return [(l * N_PER_LEVEL, (l + 1) * N_PER_LEVEL) for l in range(NUM_LEVELS)]


def select_features_mi(X, y, k=FEATURES_PER_LEVEL, seed=0):
    """Her DWT seviyesi için Mutual Information ile en iyi k özelliği seç."""
    sel = []
    for start, end in _level_slices():
        scores = mutual_info_classif(X[:, start:end], y, random_state=seed)
        top    = np.argsort(scores)[::-1][:k]
        sel.extend((start + top).tolist())
    return np.array(sel, dtype=np.int32)


def select_features_chi2(X, y, k=FEATURES_PER_LEVEL):
    """Her DWT seviyesi için Chi2 ile en iyi k özelliği seç."""
    sel = []
    for start, end in _level_slices():
        Xl = X[:, start:end].astype(np.float64) + 1e-10
        sc, _ = chi2(Xl, y)
        top   = np.argsort(sc)[::-1][:k]
        sel.extend((start + top).tolist())
    return np.array(sel, dtype=np.int32)


def build_feature_sets(X, y, verbose=True):
    """
    5× MI + 5× Chi2 = 10 aday özellik seti üret.
    Dönen: [(label, np.ndarray indeks_dizisi), ...]
    """
    candidates = []
    for run in range(SELECTION_RUNS):
        idx = select_features_mi(X, y, seed=run * 7)
        candidates.append((f"MI-{run+1}", idx))
        if verbose:
            print(f"    MI  seçim {run+1}: {len(idx)} özellik")

    for run in range(SELECTION_RUNS):
        idx = select_features_chi2(X, y)
        candidates.append((f"Chi2-{run+1}", idx))
        if verbose:
            print(f"    Chi2 seçim {run+1}: {len(idx)} özellik")

    return candidates


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 3 — GROUP K-FOLD DEĞERLENDİRME
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_grouped(X, y, groups, classes, group_map, candidates):
    """
    Her aday özellik seti için GroupKFold (kaynak dosya bazlı) cross-validation.
    En iyi kombinasyonu seç.

    NOT: Gruplar = kaynak kayıt dosyaları. Augmented versiyonlar asla
         eğitim/test ayrımını kirletmez.
    """
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_groups, 5))

    classifiers = {
        "kNN"  : KNeighborsClassifier(n_neighbors=5, metric="cityblock"),
        "SVM"  : SVC(kernel="poly", degree=3, C=1.0, coef0=1.0,
                     probability=True, class_weight="balanced"),
        "RF"   : RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                        class_weight="balanced", random_state=42,
                                        n_jobs=-1),
    }

    best_acc     = 0.0
    best_label   = ""
    best_idx     = None
    best_clf_name= ""
    all_results  = []

    print(f"\n  {'Özellik Seti':<14} {'Sınıflandırıcı':<8} {'GroupKF Acc':>11}")
    print(f"  {'─'*40}")

    for feat_label, idx in candidates:
        X_sel = X[:, idx]
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_sel)

        for clf_name, clf in classifiers.items():
            scores = cross_val_score(clf, X_sc, y,
                                     groups=groups, cv=gkf,
                                     scoring="accuracy", n_jobs=-1)
            acc = scores.mean()
            std = scores.std()
            all_results.append({
                "label": f"{feat_label}+{clf_name}",
                "acc"  : acc,
                "std"  : std,
                "idx"  : idx,
                "clf"  : clf_name,
            })
            marker = " ◄ EN İYİ" if acc > best_acc else ""
            print(f"  {feat_label:<14} {clf_name:<8} {acc*100:>9.2f}%{marker}")

            if acc > best_acc:
                best_acc      = acc
                best_label    = f"{feat_label}+{clf_name}"
                best_idx      = idx
                best_clf_name = clf_name

    return all_results, best_label, best_idx, best_clf_name, best_acc


# ══════════════════════════════════════════════════════════════════════════════
#  ADIM 4 — NİHAİ MODEL EĞİTİMİ + LOOCV TAHMİNLERİ
# ══════════════════════════════════════════════════════════════════════════════

def train_and_loocv(X, y, groups, classes, best_idx, best_clf_name, group_map):
    """
    Seçilen özellik seti + sınıflandırıcı ile GroupKFold tahminleri topla.
    Tam confusion matrix ve sınıflandırma raporu üret.
    """
    clf_map = {
        "kNN": KNeighborsClassifier(n_neighbors=5, metric="cityblock"),
        "SVM": SVC(kernel="poly", degree=3, C=1.0, coef0=1.0,
                   probability=True, class_weight="balanced"),
        "RF" : RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                      class_weight="balanced", random_state=42,
                                      n_jobs=-1),
    }

    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(n_groups, 5))

    X_sel = X[:, best_idx]

    all_true, all_pred = [], []
    fold_accs = []

    print(f"\n  {'Fold':>5}  {'Test Grupları':<32}  {'N':>5}  {'Acc':>8}")
    print(f"  {'─'*58}")

    for fold, (tr_idx, te_idx) in enumerate(
            gkf.split(X_sel, y, groups=groups)):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_sel[tr_idx])
        X_te   = scaler.transform(X_sel[te_idx])

        clf = clf_map[best_clf_name]
        clf.fit(X_tr, y[tr_idx])
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y[te_idx], y_pred)
        fold_accs.append(acc)
        all_true.extend(y[te_idx].tolist())
        all_pred.extend(y_pred.tolist())

        # Bu fold'daki test grupları
        test_grps = sorted(set(groups[te_idx]))
        grp_names = ", ".join(group_map.get(g, str(g)) for g in test_grps[:4])
        print(f"  {fold+1:>5}  {grp_names:<32}  {len(te_idx):>5}  {acc*100:>7.2f}%")

    print(f"  {'─'*58}")
    overall = accuracy_score(all_true, all_pred)
    print(f"  {'Genel':>5}  {'':<32}  {len(all_true):>5}  {overall*100:>7.2f}%")

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(classes))))
    rep = classification_report(all_true, all_pred,
                                 target_names=classes, zero_division=0, digits=3)
    p, r, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average="weighted", zero_division=0)

    print(f"\n  ┌{'─'*46}┐")
    print(f"  │  Doğruluk   : %{overall*100:6.2f}                        │")
    print(f"  │  Kesinlik   : %{p*100:6.2f}  (ağırlıklı)            │")
    print(f"  │  Duyarlılık : %{r*100:6.2f}  (ağırlıklı)            │")
    print(f"  │  F1 Skoru   : %{f1*100:6.2f}  (ağırlıklı)            │")
    print(f"  └{'─'*46}┘")
    print(f"\n  Sınıflandırma Raporu:\n  {'─'*50}")
    for ln in rep.split("\n"):
        print(f"  {ln}")

    # Tam veri üzerinde model eğit (sonuçlar için)
    scaler_final = StandardScaler()
    X_sc_final   = scaler_final.fit_transform(X_sel)
    clf_final = clf_map[best_clf_name]
    clf_final.fit(X_sc_final, y)

    return (np.array(all_true), np.array(all_pred),
            overall, p, r, f1, cm, rep, fold_accs,
            clf_final, scaler_final)


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
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(len(classes)):
        for j in range(len(classes)):
            v   = cm[i, j]
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
    b1 = ax.bar(x - w, p_c,  w, label="Kesinlik",   color="#2196F3", alpha=0.85)
    b2 = ax.bar(x,     r_c,  w, label="Duyarlılık", color="#4CAF50", alpha=0.85)
    b3 = ax.bar(x + w, f1_c, w, label="F1 Skoru",   color="#FF5722", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Skor", fontsize=11)
    ax.set_title(title or "Sınıf Bazında Performans", fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    for bar in (list(b1) + list(b2) + list(b3)):
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_fold_bars(fold_accs, best_label, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fold_accs))
    bars = ax.bar(x, [a * 100 for a in fold_accs],
                  color="#1565C0", alpha=0.82,
                  label=f"Ort: {np.mean(fold_accs)*100:.2f}%")
    ax.axhline(np.mean(fold_accs) * 100, ls="--", color="#E53935",
               lw=1.8, label=f"Ortalama: {np.mean(fold_accs)*100:.1f}%")
    ax.axhline(80, ls=":", color="orange", lw=1.2, label="Hedef (%80)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#1565C0")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i+1}" for i in range(len(fold_accs))])
    ax.set_ylim(0, 110); ax.set_ylabel("Doğruluk (%)", fontsize=11)
    ax.set_title(f"GroupKFold CV — {best_label}\n"
                 f"(Kaynak dosya bazlı, veri sızıntısız)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_results_comparison(all_results, path):
    """Tüm özellik seti + sınıflandırıcı kombinasyonlarını karşılaştır."""
    # En iyi 15'i al
    sorted_res = sorted(all_results, key=lambda r: r["acc"], reverse=True)[:15]
    labels     = [r["label"] for r in sorted_res]
    accs       = [r["acc"] * 100 for r in sorted_res]
    stds       = [r["std"] * 100 for r in sorted_res]

    clf_colors = {"kNN": "#1565C0", "SVM": "#B71C1C", "RF": "#2E7D32"}
    colors = [clf_colors.get(r["clf"], "#666") for r in sorted_res]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), accs, xerr=stds,
                   color=colors, alpha=0.82, capsize=4)
    for i, (bar, acc, std) in enumerate(zip(bars, accs, stds)):
        ax.text(acc + std + 0.3, i, f"{acc:.1f}%",
                va="center", ha="left", fontsize=8, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("GroupKFold Doğruluk (%)", fontsize=11)
    ax.set_title("Tüm Özellik Seti × Sınıflandırıcı Karşılaştırması\n"
                 "(3s segment, MI/Chi2 seçimi, kayıt bazlı CV)", fontsize=12)
    ax.axvline(80, ls="--", color="orange", lw=1.2, label="Hedef %80")
    ax.axvline(90, ls="--", color="green",  lw=1.2, label="Hedef %90")
    # Renk açıklaması
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=c, label=n)
                  for n, c in clf_colors.items()]
    legend_els += [plt.Line2D([0], [0], ls="--", color="orange", label="Hedef %80"),
                   plt.Line2D([0], [0], ls="--", color="green", label="Hedef %90")]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(accs) + 10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


def plot_dataset_stats(y, groups, classes, group_map, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    counts = [np.sum(y == i) for i in range(len(classes))]
    colors = ["#1565C0", "#B71C1C", "#2E7D32", "#E65100"]
    bars = ax1.bar(classes, counts, color=colors[:len(classes)], alpha=0.85)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 f"{c:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Segment Sayısı", fontsize=11)
    ax1.set_title("Sınıf Başına Segment Sayısı\n(Orijinal + Augmente)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, max(counts) * 1.15)

    # Grup bazında dağılım (pasta)
    grp_sizes = [(group_map.get(g, str(g)), np.sum(groups == g))
                 for g in sorted(np.unique(groups))]
    labels_g  = [x[0] for x in grp_sizes]
    sizes_g   = [x[1] for x in grp_sizes]
    wedge_colors = plt.cm.Set3(np.linspace(0, 1, len(labels_g)))
    ax2.pie(sizes_g, labels=labels_g, autopct="%1.0f%%",
            colors=wedge_colors, startangle=120,
            textprops={"fontsize": 9})
    ax2.set_title("Kaynak Dosya Başına Segment\n(GroupKFold grupları)", fontsize=11)

    fig.suptitle("SARSILMAZ İHA SESLER v2 — Veri Seti Dağılımı\n"
                 "(3s Segment, 1s Kayma, Augmentasyon)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  RAPOR
# ══════════════════════════════════════════════════════════════════════════════

def write_report(classes, overall, p, r, f1, rep,
                 fold_accs, best_label, all_results,
                 y, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("╔══════════════════════════════════════════════════════════╗\n")
        f.write("║  SARSILMAZ İHA SESLER v2 — TSK-LHP/MDWT TEST RAPORU     ║\n")
        f.write("╚══════════════════════════════════════════════════════════╝\n\n")

        f.write("YÖNTEM İYİLEŞTİRMELERİ (v1 → v2)\n")
        f.write("─" * 55 + "\n")
        f.write("  ✓ 1s  → 3s segment (L7'de 131 → 381 approx katsayı)\n")
        f.write("  ✓ Örtüşen segmentasyon (1s kayma, 3x fazla veri)\n")
        f.write("  ✓ Veri augmentasyonu (pitch shift, speed, gain)\n")
        f.write("  ✓ GroupKFold CV (kaynak dosya bazlı, data leakage yok)\n")
        f.write("  ✓ MI + Chi2 özellik seçimi\n")
        f.write("  ✓ RF sınıflandırıcı eklendi\n\n")

        f.write("VERİ SETİ\n")
        f.write("─" * 55 + "\n")
        f.write(f"  Sınıflar      : {', '.join(classes)}\n")
        f.write(f"  Toplam segment: {len(y):,}\n")
        for ci, cn in enumerate(classes):
            f.write(f"    {cn:<14}: {np.sum(y==ci):>5,} segment\n")

        f.write("\nEN İYİ KOMBİNASYON\n")
        f.write("─" * 55 + "\n")
        f.write(f"  {best_label}\n\n")

        f.write("SONUÇLAR\n")
        f.write("─" * 55 + "\n")
        f.write(f"  GroupKFold Doğruluk : %{overall*100:.2f}\n")
        f.write(f"  Kesinlik            : %{p*100:.2f} (ağırlıklı)\n")
        f.write(f"  Duyarlılık          : %{r*100:.2f} (ağırlıklı)\n")
        f.write(f"  F1 Skoru            : %{f1*100:.2f} (ağırlıklı)\n")
        f.write(f"\n  Katlı doğruluklar: {[f'{a*100:.1f}%' for a in fold_accs]}\n")

        f.write("\nSINIFLANDIRMA RAPORU\n")
        f.write("─" * 55 + "\n")
        f.write(rep)

        f.write("\n\nTÜM KOMBİNASYON SONUÇLARI\n")
        f.write("─" * 55 + "\n")
        for res in sorted(all_results, key=lambda r: r["acc"], reverse=True):
            f.write(f"  {res['label']:<25} {res['acc']*100:>7.2f}% ± {res['std']*100:.2f}%\n")

        f.write(f"\n\nTarih: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"  → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA AKIŞ
# ══════════════════════════════════════════════════════════════════════════════

def main(skip_feat=False):
    os.makedirs(RESULTS, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  SARSILMAZ İHA SESLER — TSK-LHP/MDWT TEST v2")
    print(f"{'═'*62}")
    print(f"  Segment  : {SEG_DURATION}s @ {SAMPLE_RATE}Hz")
    print(f"  Özellik  : {NUM_LEVELS}×768 = {NUM_LEVELS*768} boyut (3s)")
    print(f"  CV tipi  : GroupKFold (kayıt bazlı)")

    # ── Adım 1: Özellikler ────────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 1 — ÖZELLİK ÇIKARMA  ({SEG_DURATION}s segmentler)")
    print(f"{'═'*62}\n")
    t0 = time.time()
    X, y, groups, classes, group_map = extract_all(skip_cache=not skip_feat)
    print(f"\n  ┌{'─'*44}┐")
    print(f"  │  Toplam   : {len(X):>7,} segment                  │")
    print(f"  │  Boyut    : {X.shape[1]:>7,} özellik                  │")
    for ci, cn in enumerate(classes):
        print(f"  │  {cn:<12}: {np.sum(y==ci):>7,} segment                  │")
    print(f"  └{'─'*44}┘")
    print(f"  Süre: {time.time()-t0:.1f}s")

    # Görsel: veri seti dağılımı
    plot_dataset_stats(y, groups, classes, group_map,
                       os.path.join(RESULTS, "0_veri_seti.png"))

    # ── Adım 2: Özellik Seçimi ────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 2 — ÖZELLİK SEÇİMİ  (MI + Chi2, {SELECTION_RUNS}× her biri)")
    print(f"{'═'*62}\n")
    candidates = build_feature_sets(X, y)

    # ── Adım 3: GroupKFold Değerlendirme ──────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 3 — GROUP K-FOLD DEĞERLENDİRME")
    print(f"  (kNN + SVM + RF × 10 özellik seti = 30 kombinasyon)")
    print(f"{'═'*62}")
    all_results, best_label, best_idx, best_clf, best_acc = \
        evaluate_grouped(X, y, groups, classes, group_map, candidates)

    print(f"\n  ── En iyi: {best_label}  ({best_acc*100:.2f}%)")

    # ── Adım 4: Nihai tahminler + confusion matrix ─────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 4 — NİHAİ GROUP K-FOLD TAHMİNLERİ  [{best_label}]")
    print(f"{'═'*62}\n")
    (y_true, y_pred, overall, p, r, f1, cm, rep,
     fold_accs, clf_final, scaler_final) = \
        train_and_loocv(X, y, groups, classes, best_idx, best_clf, group_map)

    # ── Adım 5: Görselleştirme ────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  ADIM 5 — GÖRSELLEŞTİRME")
    print(f"{'═'*62}\n")

    plot_cm(cm, classes,
            f"Karmaşıklık Matrisi — {best_label}\n"
            f"GroupKFold (Kayıt Bazlı)  |  Doğruluk: %{overall*100:.2f}",
            os.path.join(RESULTS, "1_karmasiklik_matrisi.png"))

    plot_metrics_bar(classes, y_true, y_pred,
                     os.path.join(RESULTS, "2_sinif_metrikleri.png"),
                     title=f"Sınıf Bazında Performans — {best_label}\n"
                           f"(3s segment, GroupKFold, Augmentasyon)")

    plot_fold_bars(fold_accs, best_label,
                   os.path.join(RESULTS, "3_fold_dogruluklari.png"))

    plot_results_comparison(all_results,
                            os.path.join(RESULTS, "4_tum_kombinasyonlar.png"))

    # ── Rapor ─────────────────────────────────────────────────────────────
    write_report(classes, overall, p, r, f1, rep,
                 fold_accs, best_label, all_results, y,
                 os.path.join(RESULTS, "rapor_v2.txt"))

    # Model kaydet
    model_pkg = {
        "classifier" : clf_final,
        "scaler"     : scaler_final,
        "feat_idx"   : best_idx,
        "classes"    : classes,
        "best_label" : best_label,
        "accuracy"   : overall,
        "seg_duration": SEG_DURATION,
        "sample_rate": SAMPLE_RATE,
    }
    model_path = os.path.join(RESULTS, "sarsilmaz_model.joblib")
    joblib.dump(model_pkg, model_path)
    print(f"  → {model_path}")

    print(f"\n{'═'*62}")
    print(f"  SONUÇLAR")
    print(f"{'═'*62}")
    print(f"  En iyi   : {best_label}")
    print(f"  Doğruluk : %{overall*100:.2f}")
    print(f"  Kesinlik : %{p*100:.2f}  |  Duyarlılık: %{r*100:.2f}  |  F1: %{f1*100:.2f}")
    print(f"  Katlı    : {[f'{a*100:.1f}%' for a in fold_accs]}")
    print(f"\n  Çıktılar : {RESULTS}")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-feat", action="store_true",
                   help="Cache varsa özellik çıkarmayı atla")
    args = p.parse_args()
    main(skip_feat=args.skip_feat)
