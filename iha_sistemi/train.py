bana bu """
IHA Ses Sınıflandırma Sistemi - Eğitim Pipeline'ı

Kullanım:
    python train.py
    python train.py --dataset /yol/veriseti --output modeller/

Klasör yapısı:
    dataset/
    ├── IHA_Tipi_1/        ← Her alt klasör bir sınıftır
    │   ├── ses1.flac
    │   └── ses2.wav
    ├── IHA_Tipi_2/
    └── IHA_Tipi_3/
"""

import os
import glob
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings("ignore")

from config import (
    DATASET_DIR, MODEL_DIR, EXCEL_DIR,
    SAMPLE_RATE, SEGMENT_DURATION,
    NUM_LEVELS, WAVELET, NUM_BINS,
    FEATURES_PER_LEVEL, SELECTION_RUNS,
    KNN_K, KNN_METRIC, SVM_KERNEL, SVM_DEGREE, SVM_C, CV_FOLDS,
)
from feature_extractor import extract_features, load_audio


# ─── Yardımcı Sınıf: Çıkarım için özellik dönüştürücü ───────────────────────

class FeatureTransformer:
    """
    Eğitimde seçilen özellikleri çıkarım zamanında aynı şekilde uygular.
    Joblib ile serialize edilebilir.
    """

    def __init__(self, mode: str, **kwargs):
        """
        mode = 'index'  : kwargs['indices'] ile tek indeks seçimi
        mode = 'voted'  : kwargs['index_list'] ile çoklu set ortalaması
        """
        self.mode = mode
        self.kwargs = kwargs

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X : (N, 5376) tam özellik matrisi
        Döner: (N, k) seçilmiş özellik matrisi
        """
        if self.mode == "index":
            return X[:, self.kwargs["indices"]]
        elif self.mode == "voted":
            parts = np.stack(
                [X[:, idx] for idx in self.kwargs["index_list"]], axis=2
            )
            return np.mean(parts, axis=2)   # (N, k)
        else:
            raise ValueError(f"Bilinmeyen mod: {self.mode}")

    def transform_single(self, x: np.ndarray) -> np.ndarray:
        """Tek örnek için: x → (5376,), döner (k,)"""
        return self.transform(x.reshape(1, -1)).ravel()


# ─── Veri Seti Yükleme ───────────────────────────────────────────────────────

def segment_audio(audio: np.ndarray, sr: int, seg_dur: float = SEGMENT_DURATION) -> list[np.ndarray]:
    """Sesi örtüşmeyen seg_dur saniyelik segmentlere böler."""
    seg_len = int(sr * seg_dur)
    return [audio[s:s + seg_len] for s in range(0, len(audio) - seg_len + 1, seg_len)]


def build_dataset(dataset_dir: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    dataset_dir altındaki alt klasörleri tarar (her biri bir sınıf).
    Ses dosyalarını yükler, segmentler, özellik çıkarır.

    Returns:
        X       : (N, 5376) özellik matrisi
        y       : (N,) integer etiketler
        classes : sınıf adları listesi
    """
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if not classes:
        raise FileNotFoundError(
            f"Sınıf klasörü bulunamadı: {dataset_dir}\n"
            "Her IHA tipi için ayrı bir alt klasör oluşturun."
        )

    print(f"\nBulunan sınıflar ({len(classes)}): {classes}\n")

    X_list, y_list = [], []

    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(dataset_dir, class_name)
        audio_files = []
        for ext in ("*.flac", "*.wav", "*.mp3", "*.ogg", "*.FLAC", "*.WAV"):
            audio_files.extend(
                glob.glob(os.path.join(class_dir, "**", ext), recursive=True)
            )

        if not audio_files:
            print(f"  [UYARI] {class_name}: Ses dosyası bulunamadı, atlanıyor.")
            continue

        count_before = len(X_list)
        print(f"  [{class_name}] {len(audio_files)} dosya işleniyor...")

        for fpath in audio_files:
            try:
                audio, sr = load_audio(fpath, SAMPLE_RATE)
                for seg in segment_audio(audio, sr):
                    feat = extract_features(seg)
                    if feat is not None:
                        X_list.append(feat)
                        y_list.append(class_idx)
            except Exception as e:
                print(f"    HATA ({Path(fpath).name}): {e}")

        added = len(X_list) - count_before
        print(f"    → {added} segment eklendi.")

    if not X_list:
        raise RuntimeError("Hiçbir özellik çıkarılamadı!")

    return np.array(X_list), np.array(y_list, dtype=np.int32), classes


# ─── Özellik Seçimi ──────────────────────────────────────────────────────────

def _level_slices(num_levels: int = NUM_LEVELS, num_bins: int = NUM_BINS) -> list[tuple[int, int]]:
    """Her DWT seviyesinin özellik vektöründeki başlangıç/bitiş indekslerini döner."""
    n_per_level = num_bins * 3   # 768
    return [(lvl * n_per_level, (lvl + 1) * n_per_level) for lvl in range(num_levels)]


def select_by_mutual_info(X: np.ndarray, y: np.ndarray,
                           k: int = FEATURES_PER_LEVEL,
                           seed: int = 0) -> np.ndarray:
    """
    Her DWT seviyesi için mutual information ile en önemli k özelliği seçer.
    MATLAB fscnca() ile benzer amaç: bilgi teorik önem ölçütü.

    Returns: Global indeks dizisi (num_levels * k,)
    """
    selected = []
    for start, end in _level_slices():
        X_lvl = X[:, start:end]
        scores = mutual_info_classif(X_lvl, y, random_state=seed)
        top_k = np.argsort(scores)[::-1][:k]
        selected.extend((start + top_k).tolist())
    return np.array(selected, dtype=np.int32)


def select_by_chi2(X: np.ndarray, y: np.ndarray,
                   k: int = FEATURES_PER_LEVEL) -> np.ndarray:
    """
    Her DWT seviyesi için Chi-squared istatistiği ile en önemli k özelliği seçer.
    Histogram değerleri >= 0 olduğundan chi2 doğrudan uygulanabilir.

    Returns: Global indeks dizisi (num_levels * k,)
    """
    selected = []
    for start, end in _level_slices():
        X_lvl = X[:, start:end]
        # Sıfır varyans sütunları chi2'yi bozabilir — küçük epsilon ekle
        X_lvl = X_lvl + 1e-10
        scores, _ = chi2(X_lvl, y)
        top_k = np.argsort(scores)[::-1][:k]
        selected.extend((start + top_k).tolist())
    return np.array(selected, dtype=np.int32)


# ─── Değerlendirme ───────────────────────────────────────────────────────────

def cv_accuracy(clf, X: np.ndarray, y: np.ndarray, folds: int = CV_FOLDS) -> float:
    """Stratified K-Fold çapraz doğrulama ile ortalama doğruluk."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())


# ─── Ana Eğitim Fonksiyonu ───────────────────────────────────────────────────

def train(dataset_dir: str = DATASET_DIR,
          model_dir: str = MODEL_DIR,
          excel_dir: str = EXCEL_DIR) -> dict:

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 65)
    print(" ADIM 1 — Özellik Çıkarımı")
    print("=" * 65)

    X, y, classes = build_dataset(dataset_dir)
    n_samples, n_feats = X.shape
    print(f"\nToplam: {n_samples} örnek  |  Özellik boyutu: {n_feats}")
    print(f"Sınıflar ({len(classes)}): {classes}")

    # Ham özellikleri Excel'e kaydet
    cols = [f"L{(i//(NUM_BINS*3))+1}_{'S' if (i%(NUM_BINS*3))<256 else 'U' if (i%(NUM_BINS*3))<512 else 'L'}_{i%NUM_BINS}"
            for i in range(n_feats)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "label", [classes[l] for l in y])
    excel_path = os.path.join(excel_dir, "TSK_LHP_MDWT_Ozellikler.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Ham özellikler kaydedildi → {excel_path}")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(" ADIM 2 — Özellik Seçimi  (5× Mutual-Info + 5× Chi-Squared)")
    print("=" * 65)

    candidate_transformers = []   # (FeatureTransformer, np.ndarray X_sel)

    # 5 kez Mutual Information (MATLAB fscnca muadili)
    for run in range(SELECTION_RUNS):
        idx = select_by_mutual_info(X, y, seed=run * 7)
        tr  = FeatureTransformer(mode="index", indices=idx)
        candidate_transformers.append((tr, X[:, idx]))
        print(f"  MI  seçim {run+1}: {len(idx)} özellik seçildi.")

    # 5 kez Chi-Squared
    for run in range(SELECTION_RUNS):
        idx = select_by_chi2(X, y)
        tr  = FeatureTransformer(mode="index", indices=idx)
        candidate_transformers.append((tr, X[:, idx]))
        print(f"  Chi2 seçim {run+1}: {len(idx)} özellik seçildi.")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(" ADIM 3 — Sınıflandırma  (kNN + SVM, 10-Katlı CV)")
    print("=" * 65)

    # Her aday set için kNN ve SVM değerlendirmesi
    results = []   # [{'label', 'acc', 'clf', 'transformer', 'scaler'}]

    for i, (tr, X_sel) in enumerate(candidate_transformers):
        set_no = i + 1
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_sel)

        # kNN
        knn = KNeighborsClassifier(n_neighbors=KNN_K, metric=KNN_METRIC)
        acc_knn = cv_accuracy(knn, X_sc, y)
        results.append(dict(label=f"C{set_no}_kNN", acc=acc_knn,
                            clf=KNeighborsClassifier(n_neighbors=KNN_K, metric=KNN_METRIC),
                            transformer=tr, scaler=scaler))
        print(f"  S{set_no:02d} kNN  → {acc_knn*100:.2f}%")

        # SVM (çok sınıflı: OvO otomatik)
        svm = SVC(kernel=SVM_KERNEL, degree=SVM_DEGREE, C=SVM_C,
                  coef0=1.0, probability=True)
        acc_svm = cv_accuracy(svm, X_sc, y)
        results.append(dict(label=f"C{set_no}_SVM", acc=acc_svm,
                            clf=SVC(kernel=SVM_KERNEL, degree=SVM_DEGREE, C=SVM_C,
                                    coef0=1.0, probability=True),
                            transformer=tr, scaler=scaler))
        print(f"  S{set_no:02d} SVM  → {acc_svm*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(" ADIM 4 — IMV Birleştirme  (18 oylama seti, v1–v18)")
    print("=" * 65)

    np.random.seed(42)
    n_candidates = len(candidate_transformers)

    for vote_idx in range(18):
        # 4 rastgele aday set seç
        chosen = np.random.choice(n_candidates, size=4, replace=False)
        idx_sets = [candidate_transformers[c][0].kwargs["indices"] for c in chosen]

        # Her aday setin seçtiği özellikleri al ve ortalama al
        parts = np.stack([X[:, idx] for idx in idx_sets], axis=2)  # (N, k, 4)
        X_voted = np.mean(parts, axis=2)                             # (N, k)

        tr_voted = FeatureTransformer(mode="voted", index_list=idx_sets)
        scaler   = StandardScaler()
        X_sc     = scaler.fit_transform(X_voted)

        acc = cv_accuracy(
            KNeighborsClassifier(n_neighbors=KNN_K, metric=KNN_METRIC), X_sc, y
        )
        results.append(dict(
            label=f"v{vote_idx+1}", acc=acc,
            clf=KNeighborsClassifier(n_neighbors=KNN_K, metric=KNN_METRIC),
            transformer=tr_voted, scaler=scaler,
        ))
        print(f"  v{vote_idx+1:02d}      → {acc*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(" ADIM 5 — Greedy Seçim")
    print("=" * 65)

    best = max(results, key=lambda r: r["acc"])
    print(f"\n  En iyi: {best['label']}  →  {best['acc']*100:.2f}%")

    # Tam veri üzerinde son modeli eğit
    X_best_sel = best["transformer"].transform(X)
    X_best_sc  = best["scaler"].fit_transform(X_best_sel)   # Scaler'ı tüm veri ile yeniden fit et
    best["clf"].fit(X_best_sc, y)

    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print(" ADIM 6 — Kaydetme")
    print("=" * 65)

    model_package = {
        "classifier":   best["clf"],
        "transformer":  best["transformer"],
        "scaler":       best["scaler"],
        "classes":      classes,
        "best_label":   best["label"],
        "best_accuracy": best["acc"],
        "sample_rate":  SAMPLE_RATE,
        "num_levels":   NUM_LEVELS,
        "wavelet":      WAVELET,
        "all_results":  [(r["label"], round(r["acc"] * 100, 2)) for r in results],
    }

    model_path = os.path.join(model_dir, "iha_model.joblib")
    joblib.dump(model_package, model_path)
    print(f"  Model kaydedildi → {model_path}")

    # Sonuç tablosunu Excel'e kaydet
    df_res = pd.DataFrame(
        [(r["label"], f"{r['acc']*100:.2f}%") for r in sorted(results, key=lambda x: x["acc"], reverse=True)],
        columns=["Yöntem", "Doğruluk"]
    )
    res_path = os.path.join(excel_dir, "Sonuc_Karsilastirma.xlsx")
    df_res.to_excel(res_path, index=False)
    print(f"  Sonuç tablosu kaydedildi → {res_path}")

    elapsed = time.time() - t0
    print(f"\n  Toplam eğitim süresi: {elapsed/60:.1f} dakika")
    print("\n" + "=" * 65)
    print(f"  TAMAMLANDI  |  En iyi doğruluk: {best['acc']*100:.2f}%")
    print("=" * 65)

    return model_package


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IHA Ses Sınıflandırıcı — Eğitim")
    parser.add_argument("--dataset", default=DATASET_DIR,
                        help="Veri seti ana klasörü (varsayılan: config'den)")
    parser.add_argument("--output",  default=MODEL_DIR,
                        help="Model çıktı klasörü (varsayılan: config'den)")
    args = parser.parse_args()

    train(dataset_dir=args.dataset, model_dir=args.output)
