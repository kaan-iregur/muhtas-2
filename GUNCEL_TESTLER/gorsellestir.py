"""
Görselleştirme Pipeline — MDWT+TSK-LHP Sonuçları
==================================================
Her veri seti için:
  1. Karmaşıklık matrisi (CV en iyi model)
  2. Karmaşıklık matrisi (Split en iyi model)
  3. kNN/SVM/RF doğruluk karşılaştırması (CV vs Split)
  4. Sınıf bazlı F1 skorları

Tüm veri setleri karşılaştırması:
  5. CV doğruluk karşılaştırması (3 dataset × 3 model)
  6. Split doğruluk karşılaştırması
  7. CV vs Split uçurum analizi
  8. Genel özet heatmap
"""

import sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, f1_score,
                              precision_recall_fscore_support)
from sklearn.feature_selection import mutual_info_classif, chi2

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "iha_sistemi"))

# ── Yollar ────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent
OUT_S  = ROOT / "01_sentetik_5sinif"  / "sonuclar"
OUT_SR = ROOT / "02_sarsilmaz_4sinif"  / "sonuclar"
OUT_SA = ROOT / "03_saraalemadi_3sinif" / "sonuclar"
OUT_CMP = ROOT / "karsilastirma"
OUT_CMP.mkdir(exist_ok=True)

# ── Renk paleti ───────────────────────────────────────────────────────────────
NAVY    = "#1B2A4A"
BLUE    = "#2563EB"
GREEN   = "#16A34A"
ORANGE  = "#D97706"
RED     = "#DC2626"
LGRAY   = "#F3F4F6"
MGRAY   = "#9CA3AF"

MODEL_COLORS = {"kNN": BLUE, "SVM": ORANGE, "RF": GREEN}
DATASET_COLORS = {
    "Sentetik\n(5 sınıf)":    "#2563EB",
    "Sarsilmaz\n(4 sınıf)":   "#7C3AED",
    "Saraalemadi\n(3 sınıf)": "#059669",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   NAVY,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.6,
    "legend.framealpha": 0.92,
    "figure.dpi":        130,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

# ══════════════════════════════════════════════════════════════════════════════
#  VERİ YÜKLEYİCİ
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(name):
    if name == "sentetik":
        tr = np.load(ROOT / "01_sentetik_5sinif" / "cache_train.npz", allow_pickle=True)
        te = np.load(ROOT / "01_sentetik_5sinif" / "cache_test.npz",  allow_pickle=True)
        X_tr, y_tr = tr["X"].astype(np.float32), tr["y"].astype(np.int32)
        X_te, y_te = te["X"].astype(np.float32), te["y"].astype(np.int32)
        classes    = [str(c) for c in tr["classes"]]
        return X_tr, y_tr, X_te, y_te, classes, "split"
    elif name == "sarsilmaz":
        tr = np.load(Path(__file__).parent.parent / "test-2" / "cache_v3_train.npz", allow_pickle=True)
        te = np.load(Path(__file__).parent.parent / "test-2" / "cache_v3_test.npz",  allow_pickle=True)
        X_tr, y_tr = tr["X"].astype(np.float32), tr["y"].astype(np.int32)
        X_te, y_te = te["X"].astype(np.float32), te["y"].astype(np.int32)
        classes    = [str(c) for c in tr["classes"]]
        return X_tr, y_tr, X_te, y_te, classes, "temporal"
    elif name == "saraalemadi":
        d = np.load(ROOT / "03_saraalemadi_3sinif" / "cache.npz", allow_pickle=True)
        X, y  = d["X"].astype(np.float32), d["y"].astype(np.int32)
        classes = [str(c) for c in d["classes"]]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y)
        return X_tr, y_tr, X_te, y_te, classes, "random80/20"

# ══════════════════════════════════════════════════════════════════════════════
#  ORTAK DEĞERLENDİRME
# ══════════════════════════════════════════════════════════════════════════════

NUM_LEVELS, NUM_BINS, FEATURES_PER = 7, 256, 5

def level_slices():
    n = NUM_BINS * 3
    return [(i*n, (i+1)*n) for i in range(NUM_LEVELS)]

def best_idx(X, y, seed=0):
    best_acc, best_i = 0.0, None
    knn = KNeighborsClassifier(5, metric="cityblock")
    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    for run in range(5):
        for fn in [
            lambda r=run: _mi(X, y, seed=r*7),
            lambda:       _chi2(X, y)
        ]:
            idx = fn()
            Xs  = StandardScaler().fit_transform(X[:, idx])
            acc = cross_val_score(knn, Xs, y, cv=skf,
                                  scoring="accuracy", n_jobs=-1).mean()
            if acc > best_acc:
                best_acc, best_i = acc, idx
    return best_i

def _mi(X, y, seed=0, k=FEATURES_PER):
    sel = []
    for s, e in level_slices():
        sc = mutual_info_classif(X[:, s:e], y, random_state=seed)
        sel.extend((s + np.argsort(sc)[::-1][:k]).tolist())
    return np.array(sel, np.int32)

def _chi2(X, y, k=FEATURES_PER):
    sel = []
    for s, e in level_slices():
        sc, _ = chi2(X[:, s:e] + 1e-10, y)
        sel.extend((s + np.argsort(sc)[::-1][:k]).tolist())
    return np.array(sel, np.int32)

def make_clfs():
    return [
        ("kNN", KNeighborsClassifier(5, metric="cityblock")),
        ("SVM", SVC(kernel="poly", degree=3, C=1.0, coef0=1.0,
                    probability=True, random_state=42)),
        ("RF",  RandomForestClassifier(200, random_state=42, n_jobs=-1)),
    ]

def evaluate(X_tr, y_tr, X_te, y_te, idx):
    scaler  = StandardScaler()
    Xs_tr   = scaler.fit_transform(X_tr[:, idx])
    Xs_te   = scaler.transform(X_te[:, idx])
    skf     = StratifiedKFold(10, shuffle=True, random_state=42)

    cv_res, sp_res = [], []

    for name, clf in make_clfs():
        # CV
        cv_all_pred, cv_all_true = [], []
        cv_accs = []
        for tr_i, te_i in skf.split(Xs_tr, y_tr):
            clf.fit(Xs_tr[tr_i], y_tr[tr_i])
            pred = clf.predict(Xs_tr[te_i])
            cv_all_pred.extend(pred)
            cv_all_true.extend(y_tr[te_i])
            cv_accs.append(accuracy_score(y_tr[te_i], pred))
        cv_acc = float(np.mean(cv_accs))
        cv_f1  = float(f1_score(cv_all_true, cv_all_pred, average="weighted"))
        cv_res.append({"name": name, "acc": cv_acc,
                       "acc_std": float(np.std(cv_accs)), "f1": cv_f1,
                       "pred": np.array(cv_all_pred),
                       "true": np.array(cv_all_true)})

        # Split
        clf.fit(Xs_tr, y_tr)
        sp_pred = clf.predict(Xs_te)
        sp_acc  = accuracy_score(y_te, sp_pred)
        sp_f1   = f1_score(y_te, sp_pred, average="weighted")
        sp_res.append({"name": name, "acc": sp_acc, "f1": sp_f1,
                       "pred": sp_pred, "true": y_te})

    return cv_res, sp_res

# ══════════════════════════════════════════════════════════════════════════════
#  GRAFİK FONKSİYONLARI
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion(ax, pred, true, classes, title, cmap="Blues"):
    cm  = confusion_matrix(true, pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    sns.heatmap(cm_pct, annot=False, fmt=".0f", cmap=cmap,
                xticklabels=classes, yticklabels=classes,
                linewidths=0.4, linecolor="#d1d5db",
                cbar_kws={"shrink": 0.75, "format": "%.0f%%"},
                ax=ax, vmin=0, vmax=100)
    # Hücre etiketleri: sayı + %
    for i in range(len(classes)):
        for j in range(len(classes)):
            val_n   = cm[i, j]
            val_pct = cm_pct[i, j]
            color   = "white" if val_pct > 55 else NAVY
            ax.text(j + 0.5, i + 0.5,
                    f"{val_n}\n({val_pct:.0f}%)",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=color)
    ax.set_title(title, pad=10)
    ax.set_ylabel("Gerçek Sınıf", fontsize=9)
    ax.set_xlabel("Tahmin Edilen Sınıf", fontsize=9)
    ax.tick_params(axis="x", rotation=25, labelsize=8.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=8.5)


def plot_model_comparison(ax, cv_res, sp_res, title):
    names  = [r["name"] for r in cv_res]
    cv_acc = [r["acc"] * 100 for r in cv_res]
    sp_acc = [r["acc"] * 100 for r in sp_res]
    cv_std = [r["acc_std"] * 100 for r in cv_res]

    x  = np.arange(len(names))
    w  = 0.35
    bars1 = ax.bar(x - w/2, cv_acc, w, label="10-Fold CV",
                   color=[MODEL_COLORS[n] for n in names],
                   alpha=0.85, zorder=3,
                   yerr=cv_std, capsize=4, error_kw={"linewidth": 1.2})
    bars2 = ax.bar(x + w/2, sp_acc, w, label="Bağımsız Split",
                   color=[MODEL_COLORS[n] for n in names],
                   alpha=0.40, zorder=3, hatch="///")

    # Değer etiketleri
    for bar, val in zip(bars1, cv_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"%{val:.1f}", ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color=NAVY)
    for bar, val in zip(bars2, sp_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"%{val:.1f}", ha="center", va="bottom", fontsize=8.5,
                color=MGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(max(0, min(cv_acc + sp_acc) - 12), 103)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_title(title, pad=10)
    solid = mpatches.Patch(facecolor=NAVY, alpha=0.7, label="10-Fold CV (± std)")
    hatch = mpatches.Patch(facecolor=NAVY, alpha=0.3, hatch="///", label="Bağımsız Split")
    ax.legend(handles=[solid, hatch], fontsize=8.5, loc="lower right")


def plot_class_f1(ax, pred, true, classes, title, color):
    p, r, f, _ = precision_recall_fscore_support(true, pred, labels=range(len(classes)))
    y = np.arange(len(classes))
    h = 0.25
    bars_p = ax.barh(y + h,   p * 100, h, label="Kesinlik",  color=color,  alpha=0.90)
    bars_r = ax.barh(y,       r * 100, h, label="Duyarlılık", color=color,  alpha=0.55)
    bars_f = ax.barh(y - h,   f * 100, h, label="F1",         color=color,  alpha=0.30, hatch="//")

    for bar in bars_f:
        val = bar.get_width()
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f"%{val:.1f}", va="center", fontsize=8, color=NAVY)

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Skor (%)")
    ax.set_title(title, pad=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.invert_yaxis()


# ══════════════════════════════════════════════════════════════════════════════
#  VERİ SETİ BAŞINA PANEL (4 grafik)
# ══════════════════════════════════════════════════════════════════════════════

def dataset_panel(ds_name, label, classes, cv_res, sp_res, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    best_cv = max(cv_res, key=lambda r: r["acc"])
    best_sp = max(sp_res, key=lambda r: r["acc"])

    # ── Figür 1: İki karmaşıklık matrisi yan yana ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"{label} — Karmaşıklık Matrisleri",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.01)

    plot_confusion(axes[0], best_cv["pred"], best_cv["true"], classes,
                   f"10-Fold CV — {best_cv['name']}\n"
                   f"Doğruluk: %{best_cv['acc']*100:.2f}  F1: %{best_cv['f1']*100:.2f}",
                   cmap="Blues")
    plot_confusion(axes[1], best_sp["pred"], best_sp["true"], classes,
                   f"Bağımsız Split — {best_sp['name']}\n"
                   f"Doğruluk: %{best_sp['acc']*100:.2f}  F1: %{best_sp['f1']*100:.2f}",
                   cmap="Greens")

    plt.tight_layout()
    fig.savefig(out_dir / "1_karmasiklik_matrisleri.png")
    plt.close(fig)
    print(f"    ✓ 1_karmasiklik_matrisleri.png")

    # ── Figür 2: Model karşılaştırma (CV vs Split) ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_model_comparison(ax, cv_res, sp_res,
                          f"{label}\nkNN / SVM / RF — 10-Fold CV vs Bağımsız Split")
    plt.tight_layout()
    fig.savefig(out_dir / "2_model_karsilastirma.png")
    plt.close(fig)
    print(f"    ✓ 2_model_karsilastirma.png")

    # ── Figür 3: Sınıf bazlı F1 (CV en iyi model) ─────────────────────────────
    colors = [BLUE, ORANGE, GREEN]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(classes) * 1.1 + 1.5)))
    fig.suptitle(f"{label} — Sınıf Bazlı Metrikler",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.01)

    plot_class_f1(axes[0], best_cv["pred"], best_cv["true"], classes,
                  f"10-Fold CV — {best_cv['name']}", BLUE)
    plot_class_f1(axes[1], best_sp["pred"], best_sp["true"], classes,
                  f"Bağımsız Split — {best_sp['name']}", GREEN)

    plt.tight_layout()
    fig.savefig(out_dir / "3_sinif_metrikleri.png")
    plt.close(fig)
    print(f"    ✓ 3_sinif_metrikleri.png")

    # ── Figür 4: CV fold doğrulukları (kNN/SVM/RF) ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names  = [r["name"] for r in cv_res]
    accs   = [r["acc"] * 100 for r in cv_res]
    stds   = [r["acc_std"] * 100 for r in cv_res]
    f1s    = [r["f1"] * 100 for r in cv_res]

    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, accs, w,
                color=[MODEL_COLORS[n] for n in names], alpha=0.85, zorder=3,
                yerr=stds, capsize=5, error_kw={"linewidth": 1.5},
                label="Doğruluk")
    b2 = ax.bar(x + w/2, f1s, w,
                color=[MODEL_COLORS[n] for n in names], alpha=0.38,
                hatch="xxx", zorder=3, label="F1 Skoru")

    for bar, val, std in zip(b1, accs, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.5,
                f"%{val:.1f}", ha="center", fontsize=8.5,
                fontweight="bold", color=NAVY)
    for bar, val in zip(b2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"%{val:.1f}", ha="center", fontsize=8, color=MGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(max(0, min(accs + f1s) - 15), 105)
    ax.set_ylabel("Skor (%)")
    ax.set_title(f"{label} — 10-Fold CV: Doğruluk & F1 Karşılaştırması", pad=10)
    solid = mpatches.Patch(facecolor=NAVY, alpha=0.8, label="Doğruluk (± std)")
    hatch = mpatches.Patch(facecolor=NAVY, alpha=0.3, hatch="xxx", label="F1 Skoru")
    ax.legend(handles=[solid, hatch], fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "4_cv_dogruluk_f1.png")
    plt.close(fig)
    print(f"    ✓ 4_cv_dogruluk_f1.png")

    return {
        "label": label,
        "classes": classes,
        "cv_res": cv_res,
        "sp_res": sp_res,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GENEL KARŞILAŞTIRMA GRAFİKLERİ
# ══════════════════════════════════════════════════════════════════════════════

def comparison_plots(all_results):
    ds_labels  = [r["label"] for r in all_results]
    model_names = ["kNN", "SVM", "RF"]

    cv_matrix  = np.array([[next(x["acc"]*100 for x in r["cv_res"] if x["name"]==m)
                             for m in model_names] for r in all_results])
    sp_matrix  = np.array([[next(x["acc"]*100 for x in r["sp_res"] if x["name"]==m)
                             for m in model_names] for r in all_results])
    cv_std_mat = np.array([[next(x["acc_std"]*100 for x in r["cv_res"] if x["name"]==m)
                             for m in model_names] for r in all_results])

    # ── Grafik A: Grouped bar — CV doğruluk (3 dataset, 3 model) ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Tüm Veri Setleri — Model Karşılaştırması",
                 fontsize=15, fontweight="bold", color=NAVY)

    for ax_idx, (ax, matrix, std_mat, title, alpha_vals) in enumerate([
        (axes[0], cv_matrix,  cv_std_mat, "10-Fold Cross-Validation", (0.90, 0.65, 0.40)),
        (axes[1], sp_matrix,  None,       "Bağımsız Split",           (0.90, 0.65, 0.40)),
    ]):
        x = np.arange(len(ds_labels))
        w = 0.26
        offsets = [-w, 0, w]

        for mi, (mname, offset, alpha) in enumerate(zip(model_names, offsets, alpha_vals)):
            vals  = matrix[:, mi]
            stds  = std_mat[:, mi] if std_mat is not None else np.zeros(len(vals))
            bars  = ax.bar(x + offset, vals, w,
                           label=mname, color=list(MODEL_COLORS.values())[mi],
                           alpha=alpha, zorder=3,
                           yerr=stds, capsize=4 if std_mat is not None else 0,
                           error_kw={"linewidth": 1.2})
            for di2, (bar, val, std) in enumerate(zip(bars, vals, stds)):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + std + 0.8,
                        f"%{val:.1f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color=NAVY)

        ax.set_xticks(x)
        ax.set_xticklabels(ds_labels, fontsize=10)
        ax.set_ylim(max(0, matrix.min() - 18), 105)
        ax.set_ylabel("Doğruluk (%)")
        ax.set_title(title, fontsize=12, pad=8)
        ax.legend(fontsize=9, title="Model", title_fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_CMP / "A_model_karsilastirma.png")
    plt.close(fig)
    print("  ✓ A_model_karsilastirma.png")

    # ── Grafik B: CV vs Split uçurum analizi ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))

    best_cv = cv_matrix.max(axis=1)
    best_sp = sp_matrix.max(axis=1)
    gap     = best_cv - best_sp

    x = np.arange(len(ds_labels))
    w = 0.32
    b1 = ax.bar(x - w/2, best_cv, w, label="En iyi CV",
                color=[BLUE, "#7C3AED", "#059669"], alpha=0.85, zorder=3)
    b2 = ax.bar(x + w/2, best_sp, w, label="En iyi Split",
                color=[BLUE, "#7C3AED", "#059669"], alpha=0.40, zorder=3, hatch="///")

    # Uçurum okları
    for i, (cv_v, sp_v, g) in enumerate(zip(best_cv, best_sp, gap)):
        mid_x = x[i] + w/2
        ax.annotate("", xy=(mid_x, sp_v + 0.3), xytext=(mid_x, cv_v - 0.3),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
        ax.text(mid_x + 0.04, (cv_v + sp_v) / 2, f"Δ{g:.1f}%",
                fontsize=9, color=RED, fontweight="bold", va="center")
        # Değer etiketleri
        ax.text(x[i] - w/2, cv_v + 0.6, f"%{cv_v:.1f}",
                ha="center", fontsize=8.5, fontweight="bold", color=NAVY)
        ax.text(mid_x,      sp_v + 0.6, f"%{sp_v:.1f}",
                ha="center", fontsize=8.5, color=MGRAY)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, fontsize=10)
    ax.set_ylim(max(0, min(best_sp) - 15), 105)
    ax.set_ylabel("Doğruluk (%)")
    ax.set_title("CV vs Bağımsız Split — Uçurum Analizi (Δ = Data Leakage Etkisi)",
                 pad=10)
    solid = mpatches.Patch(facecolor=NAVY, alpha=0.8, label="10-Fold CV (en iyi model)")
    hatch = mpatches.Patch(facecolor=NAVY, alpha=0.35, hatch="///", label="Bağımsız Split (en iyi model)")
    ax.legend(handles=[solid, hatch], fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_CMP / "B_cv_split_ucurum.png")
    plt.close(fig)
    print("  ✓ B_cv_split_ucurum.png")

    # ── Grafik C: Heatmap — tüm sonuçlar tek tabloda ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Genel Sonuç Özeti — Heatmap",
                 fontsize=14, fontweight="bold", color=NAVY)

    row_labels = [r["label"].replace("\n", " ") for r in all_results]

    for ax, matrix, title, fmt_str in [
        (axes[0], cv_matrix, "10-Fold CV Doğruluk (%)", ".1f"),
        (axes[1], sp_matrix, "Bağımsız Split Doğruluk (%)", ".1f"),
    ]:
        sns.heatmap(matrix, annot=True, fmt=fmt_str, cmap="YlOrRd",
                    xticklabels=model_names,
                    yticklabels=row_labels,
                    linewidths=0.8, linecolor="white",
                    cbar_kws={"shrink": 0.8},
                    vmin=60, vmax=100,
                    annot_kws={"size": 13, "weight": "bold"},
                    ax=ax)
        ax.set_title(title, pad=8, fontsize=11)
        ax.tick_params(axis="x", labelsize=11, rotation=0)
        ax.tick_params(axis="y", labelsize=10, rotation=0)
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("")

        # % işareti ekle
        for t in ax.texts:
            t.set_text(t.get_text() + "%")

    plt.tight_layout()
    fig.savefig(OUT_CMP / "C_ozet_heatmap.png")
    plt.close(fig)
    print("  ✓ C_ozet_heatmap.png")

    # ── Grafik D: Radar / spider — 3 dataset profil karşılaştırması ───────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(polar=True))
    fig.suptitle("Veri Seti Profilleri — Model Performans Radarı",
                 fontsize=14, fontweight="bold", color=NAVY)

    angles   = np.linspace(0, 2 * np.pi, len(model_names), endpoint=False).tolist()
    angles  += angles[:1]

    ds_colors = [BLUE, "#7C3AED", "#059669"]

    for di, (ax, result, color) in enumerate(zip(axes, all_results, ds_colors)):
        cv_vals  = list(cv_matrix[di]) + [cv_matrix[di][0]]
        sp_vals  = list(sp_matrix[di]) + [sp_matrix[di][0]]

        ax.plot(angles, cv_vals,  "o-", lw=2, color=color,      label="CV")
        ax.fill(angles, cv_vals, alpha=0.20, color=color)
        ax.plot(angles, sp_vals, "s--", lw=2, color=color, alpha=0.60, label="Split")
        ax.fill(angles, sp_vals, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(model_names, size=11, fontweight="bold")
        ax.set_ylim(60, 100)
        ax.set_yticks([65, 75, 85, 95])
        ax.set_yticklabels(["65%", "75%", "85%", "95%"], size=7, color=MGRAY)
        ax.set_title(result["label"].replace("\n", " "),
                     pad=15, fontsize=11, color=color, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        ax.spines["polar"].set_color("#d1d5db")
        ax.grid(color="#e5e7eb", linewidth=0.8)

    plt.tight_layout()
    fig.savefig(OUT_CMP / "D_radar_profil.png")
    plt.close(fig)
    print("  ✓ D_radar_profil.png")


# ══════════════════════════════════════════════════════════════════════════════
#  ANA PROGRAM
# ══════════════════════════════════════════════════════════════════════════════

DATASETS = [
    ("sentetik",    "Sentetik\n(5 sınıf)",    OUT_S),
    ("sarsilmaz",   "Sarsilmaz\n(4 sınıf)",   OUT_SR),
    ("saraalemadi", "Saraalemadi\n(3 sınıf)", OUT_SA),
]

if __name__ == "__main__":
    all_results = []

    for ds_key, ds_label, out_dir in DATASETS:
        print(f"\n{'='*60}")
        print(f"  {ds_label.replace(chr(10),' ')} işleniyor...")
        print(f"{'='*60}")

        X_tr, y_tr, X_te, y_te, classes, split_type = load_dataset(ds_key)
        print(f"  Özellik seçimi...")
        idx = best_idx(X_tr, y_tr)
        print(f"  Değerlendirme (CV + Split)...")
        cv_res, sp_res = evaluate(X_tr, y_tr, X_te, y_te, idx)

        print(f"  Grafikler oluşturuluyor → {out_dir}")
        result = dataset_panel(ds_key, ds_label, classes,
                               cv_res, sp_res, out_dir)
        all_results.append(result)

    print(f"\n{'='*60}")
    print("  Karşılaştırma grafikleri...")
    print(f"{'='*60}")
    comparison_plots(all_results)

    print(f"\n  TAMAMLANDI. Çıktı klasörleri:")
    print(f"    {OUT_S}")
    print(f"    {OUT_SR}")
    print(f"    {OUT_SA}")
    print(f"    {OUT_CMP}")
