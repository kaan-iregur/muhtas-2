"""
İHA Model Adaptörü — Her iki model formatını tek arayüzde yönetir.

Desteklenen formatlar:
  - Yeni (iha_sistemi/train.py): 'transformer' + 'num_levels' + 'wavelet' anahtarları
  - Eski (test-2 Sarsilmaz):     'feat_idx' anahtarı
"""

import os
import sys
import numpy as np
import joblib

from config import DEFAULT_MODEL, NUM_LEVELS, WAVELET, CONFIDENCE_THRESHOLD


def load_model(model_path: str = DEFAULT_MODEL) -> dict:
    """
    Model dosyasını yükler, format tespiti yapar.

    Returns:
        model_info dict:
          classes       : sınıf adları listesi
          sample_rate   : eğitim örnekleme Hz
          seg_duration  : pencere süresi (saniye)
          best_label    : en iyi yöntem adı
          accuracy      : CV/split doğruluğu
          _pkg          : ham joblib paketi (iç kullanım)
          _fmt          : 'new' veya 'old'
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}\n"
            "test-2/sonuclar_v3/sarsilmaz_model_v3.joblib mevcut olmalı."
        )

    # Eski format modeller FeatureTransformer sınıfına ihtiyaç duyabilir
    iha_dir = os.path.join(os.path.dirname(__file__), "..", "iha_sistemi")
    iha_dir = os.path.abspath(iha_dir)
    if os.path.isdir(iha_dir) and iha_dir not in sys.path:
        sys.path.insert(0, iha_dir)

    pkg = joblib.load(model_path)

    if "transformer" in pkg:
        fmt = "new"
    elif "feat_idx" in pkg:
        fmt = "old"
    else:
        raise ValueError(f"Tanınmayan model formatı. Anahtarlar: {list(pkg.keys())}")

    return {
        "classes":      pkg["classes"],
        "sample_rate":  pkg.get("sample_rate", 16000),
        "seg_duration": pkg.get("seg_duration", 3.0),
        "best_label":   pkg.get("best_label", "?"),
        "accuracy":     pkg.get("best_accuracy", pkg.get("accuracy", 0.0)),
        "_pkg":         pkg,
        "_fmt":         fmt,
    }


def predict(audio: np.ndarray, model_info: dict) -> tuple[str, float, dict]:
    """
    Bir ses segmentinden tahmin üretir.

    Args:
        audio       : (N,) float64, 16kHz mono, normalize edilmemiş
        model_info  : load_model() çıktısı

    Returns:
        (predicted_class, confidence, {sınıf: olasılık})
    """
    from feature_extractor import extract_features

    pkg     = model_info["_pkg"]
    fmt     = model_info["_fmt"]
    classes = model_info["classes"]

    num_levels = pkg.get("num_levels", NUM_LEVELS)
    wavelet    = pkg.get("wavelet", WAVELET)

    feat = extract_features(audio, num_levels=num_levels, wavelet=wavelet)
    if feat is None:
        return "Belirsiz", 0.0, {c: 0.0 for c in classes}

    X = feat.reshape(1, -1).astype(np.float64)

    # Özellik seçimi
    if fmt == "new":
        X_sel = pkg["transformer"].transform(X)
    else:
        X_sel = X[:, pkg["feat_idx"]]

    # Normalize + tahmin
    X_sc    = pkg["scaler"].transform(X_sel)
    clf     = pkg["classifier"]
    pred_idx = int(clf.predict(X_sc)[0])

    # Olasılık hesabı
    if hasattr(clf, "predict_proba"):
        proba       = clf.predict_proba(X_sc)[0]
        confidence  = float(proba[pred_idx])
        class_probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
    elif hasattr(clf, "decision_function"):
        df = clf.decision_function(X_sc)[0]
        if not hasattr(df, "__len__"):
            df = np.array([df, -df])
        df_exp      = np.exp(df - df.max())
        proba       = df_exp / df_exp.sum()
        confidence  = float(proba[pred_idx]) if pred_idx < len(proba) else 1.0
        n           = min(len(classes), len(proba))
        class_probs = {classes[i]: float(proba[i]) for i in range(n)}
    else:
        confidence  = 1.0
        class_probs = {classes[pred_idx]: 1.0}

    # Eksik sınıfları sıfırla tamamla
    for c in classes:
        class_probs.setdefault(c, 0.0)

    predicted_class = classes[pred_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Belirsiz"

    return predicted_class, confidence, class_probs
