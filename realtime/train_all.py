"""
Sarsilmaz 4-Sınıf — kNN + SVM + RF Model Eğitimi
===================================================
Var olan cache'den (test-2/cache_v3_*.npz) özellik matrislerini yükler,
MI özellik seçimi yapar, üç sınıflandırıcıyı eğitip ayrı ayrı kaydeder.

Çıktı:  realtime/models/model_knn.joblib
        realtime/models/model_svm.joblib
        realtime/models/model_rf.joblib

Kullanım:
  cd realtime/
  python train_all.py
"""

import os
import sys
import time
import warnings
import numpy as np
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import PROJ_DIR

# ─── Yollar ──────────────────────────────────────────────────────────────────
CACHE_TRAIN = os.path.join(PROJ_DIR, "test-2", "cache_v3_train.npz")
CACHE_TEST  = os.path.join(PROJ_DIR, "test-2", "cache_v3_test.npz")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "models")

# ─── Sabitler ────────────────────────────────────────────────────────────────
NUM_LEVELS     = 7
NUM_BINS       = 256
FEATURES_PER   = 5          # seviye başına seçilen özellik
SELECTION_RUNS = 5          # MI kaç kez tekrarlanır
CV_FOLDS       = 10
SAMPLE_RATE    = 16000
SEG_DURATION   = 3.0


# ─── Özellik seçimi ──────────────────────────────────────────────────────────

def level_slices():
    n = NUM_BINS * 3   # 768
    return [(i * n, (i + 1) * n) for i in range(NUM_LEVELS)]


def select_mi(X, y, k=FEATURES_PER, seed=0):
    sel = []
    for s, e in level_slices():
        scores = mutual_info_classif(X[:, s:e], y, random_state=seed)
        sel.extend((s + np.argsort(scores)[::-1][:k]).tolist())
    return np.array(sel, dtype=np.int32)


def best_feature_set(X_tr, y_tr):
    """
    SELECTION_RUNS × MI → her biri 10-fold kNN CV ile değerlendirilir.
    En yüksek CV doğruluğunu veren indeks seti döner.
    """
    best_acc, best_idx = 0.0, None
    ref_clf = KNeighborsClassifier(n_neighbors=5, metric="cityblock")
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42)

    for run in range(SELECTION_RUNS):
        idx = select_mi(X_tr, y_tr, seed=run * 7)
        Xs  = StandardScaler().fit_transform(X_tr[:, idx])
        acc = cross_val_score(ref_clf, Xs, y_tr, cv=skf,
                              scoring="accuracy", n_jobs=-1).mean()
        print(f"    MI-{run+1}: {len(idx)} özellik  CV={acc*100:.2f}%")
        if acc > best_acc:
            best_acc, best_idx = acc, idx

    return best_idx, best_acc


# ─── CV doğruluğu ─────────────────────────────────────────────────────────────

def cv_acc(clf, X, y):
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=skf,
                                 scoring="accuracy", n_jobs=-1).mean())


# ─── Kaydetme ────────────────────────────────────────────────────────────────

def save_model(clf, scaler, feat_idx, classes, label, cv_accuracy,
               split_accuracy, path):
    pkg = {
        "classifier":     clf,
        "scaler":         scaler,
        "feat_idx":       feat_idx,
        "classes":        classes,
        "best_label":     label,
        "cv_accuracy":    cv_accuracy,
        "accuracy":       split_accuracy,
        "seg_duration":   SEG_DURATION,
        "sample_rate":    SAMPLE_RATE,
    }
    joblib.dump(pkg, path)
    print(f"    → Kaydedildi: {path}")


# ─── Ana eğitim ──────────────────────────────────────────────────────────────

def train():
    t0 = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Veri yükleme ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  ADIM 1 — Veri Yükleme")
    print("=" * 60)

    if not os.path.exists(CACHE_TRAIN):
        raise FileNotFoundError(
            f"Cache bulunamadı: {CACHE_TRAIN}\n"
            "GUNCEL_TESTLER/unifiye_test.py --dataset sarsilmaz çalıştır."
        )

    d_tr = np.load(CACHE_TRAIN, allow_pickle=True)
    d_te = np.load(CACHE_TEST,  allow_pickle=True)
    X_tr = d_tr["X"].astype(np.float32)
    y_tr = d_tr["y"].astype(np.int32)
    X_te = d_te["X"].astype(np.float32)
    y_te = d_te["y"].astype(np.int32)
    classes = [str(c) for c in d_tr["classes"]]

    print(f"  Train: {X_tr.shape[0]} örnek  |  Test: {X_te.shape[0]} örnek")
    print(f"  Sınıflar: {classes}")
    for i, c in enumerate(classes):
        print(f"    {c}: train={(y_tr==i).sum()}  test={(y_te==i).sum()}")

    # ── 2. Özellik seçimi ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ADIM 2 — MI Özellik Seçimi (train verisi)")
    print("=" * 60)
    best_idx, ref_acc = best_feature_set(X_tr, y_tr)
    print(f"\n  Seçilen özellik sayısı : {len(best_idx)}")
    print(f"  Referans CV (kNN)      : %{ref_acc*100:.2f}")

    # Ölçekleme (train fit, test transform)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr[:, best_idx])
    X_te_sc  = scaler.transform(X_te[:, best_idx])

    # ── 3. Sınıflandırıcılar ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ADIM 3 — Sınıflandırıcı Eğitimi + Değerlendirme")
    print("=" * 60)

    classifiers = [
        ("kNN", KNeighborsClassifier(n_neighbors=5, metric="cityblock")),
        ("SVM", SVC(kernel="poly", degree=3, C=1.0, coef0=1.0,
                    probability=True, random_state=42)),
        ("RF",  RandomForestClassifier(n_estimators=200, random_state=42,
                                        n_jobs=-1)),
    ]

    model_paths = {
        "kNN": os.path.join(MODEL_DIR, "model_knn.joblib"),
        "SVM": os.path.join(MODEL_DIR, "model_svm.joblib"),
        "RF":  os.path.join(MODEL_DIR, "model_rf.joblib"),
    }

    print(f"\n  {'Model':<6}  {'10-fold CV':>12}  {'Temporal Split':>16}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*16}")

    for label, clf in classifiers:
        t1 = time.time()

        # 10-fold CV (train verisi)
        acc_cv = cv_acc(clf, X_tr_sc, y_tr)

        # Tam train üzerinde eğit, test setinde değerlendir
        clf.fit(X_tr_sc, y_tr)
        acc_split = accuracy_score(y_te, clf.predict(X_te_sc))

        elapsed_s = time.time() - t1
        print(f"  {label:<6}  {acc_cv*100:>11.2f}%  {acc_split*100:>15.2f}%"
              f"  ({elapsed_s:.0f}s)")

        # Model kaydet (scaler her model için aynı — ayrı kopyala)
        import copy
        save_model(clf, copy.deepcopy(scaler), best_idx.copy(),
                   classes, label, acc_cv, acc_split, model_paths[label])

    # ── Özet ─────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  TAMAMLANDI  |  Toplam süre: {elapsed:.0f}s")
    print(f"  Modeller: {MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    train()
