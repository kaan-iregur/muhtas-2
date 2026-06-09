"""
IHA Ses Sınıflandırma Sistemi - Gerçek Zamanlı Çıkarım
Jetson Nano Orin üzerinde mikrofon girişi ile çalışır.

Kullanım:
    python realtime.py
    python realtime.py --model modeller/iha_model.joblib --device 1
    python realtime.py --list-devices
"""

import os
import sys
import time
import argparse
import threading
import queue
import warnings
from collections import deque
from datetime import datetime

import numpy as np
import joblib
import sounddevice as sd

warnings.filterwarnings("ignore")

from config import (
    MODEL_DIR, SAMPLE_RATE, SEGMENT_DURATION, HOP_DURATION,
    MIC_DEVICE, CONFIDENCE_THRESHOLD, DISPLAY_TOP_N,
)
from feature_extractor import extract_features


# ─── Model Yükleme ───────────────────────────────────────────────────────────

def load_model(model_path: str) -> dict:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {model_path}\n"
            "Önce 'python train.py' komutunu çalıştırın."
        )
    pkg = joblib.load(model_path)
    print(f"\nModel yüklendi: {model_path}")
    print(f"  Sınıflar    : {pkg['classes']}")
    print(f"  CV Doğruluk : %{pkg['best_accuracy']*100:.2f}")
    print(f"  Yöntem      : {pkg['best_label']}")
    return pkg


# ─── Tek Örnek Tahmin ────────────────────────────────────────────────────────

def predict(audio_segment: np.ndarray, model_pkg: dict) -> tuple[str, float, dict]:
    """
    Args:
        audio_segment : (N,) float64 ses sinyali
        model_pkg     : joblib'den yüklenmiş model paketi

    Returns:
        (predicted_class, confidence, class_probabilities)
    """
    feat = extract_features(audio_segment,
                            num_levels=model_pkg["num_levels"],
                            wavelet=model_pkg["wavelet"])
    if feat is None:
        return "Belirsiz", 0.0, {}

    X = feat.reshape(1, -1).astype(np.float64)

    # Özellik seçimi
    transformer = model_pkg["transformer"]
    X_sel = transformer.transform(X)

    # Ölçekleme
    X_sc = model_pkg["scaler"].transform(X_sel)

    # Tahmin
    clf     = model_pkg["classifier"]
    classes = model_pkg["classes"]
    pred_idx = int(clf.predict(X_sc)[0])

    # Güven skoru (olasılık varsa kullan, yoksa 1/0)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_sc)[0]
        confidence = float(proba[pred_idx])
        class_probs = {classes[i]: float(proba[i]) for i in range(len(classes))}
    elif hasattr(clf, "decision_function"):
        df = clf.decision_function(X_sc)[0]
        if df.ndim == 0:
            df = np.array([df, -df])
        # Softmax normalizasyon
        df_exp = np.exp(df - df.max())
        proba  = df_exp / df_exp.sum()
        confidence  = float(proba[pred_idx]) if len(proba) > pred_idx else 1.0
        class_probs = {classes[i]: float(proba[i]) for i in range(len(classes))} \
                      if len(proba) == len(classes) else {classes[pred_idx]: 1.0}
    else:
        confidence  = 1.0
        class_probs = {classes[pred_idx]: 1.0}

    predicted_class = classes[pred_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Belirsiz"

    return predicted_class, confidence, class_probs


# ─── Gerçek Zamanlı Sistem ───────────────────────────────────────────────────

class RealTimeIHAClassifier:
    """
    Mikrofon → buffer → özellik çıkarımı → tahmin → ekrana yazdır.

    Mimarisi:
        Ses callback'i (sounddevice thread)
            → audio_queue
        İşleme thread'i
            → extract_features + predict
            → result_queue
        Gösterim (ana thread)
            → print / LED / GUI vs.
    """

    def __init__(self, model_pkg: dict, device=None):
        self.model_pkg    = model_pkg
        self.device       = device
        self.sr           = model_pkg.get("sample_rate", SAMPLE_RATE)
        self.seg_len      = int(self.sr * SEGMENT_DURATION)
        self.hop_len      = int(self.sr * HOP_DURATION)

        # Kayan ses tamponu (son 2 saniyelik)
        self.buffer       = deque(maxlen=self.seg_len * 2)
        self.samples_since_last = 0

        self.audio_queue  = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=20)

        self._running     = False
        self._proc_thread = None

        # Tahmin geçmişi (son 5 sonuç)
        self.history      = deque(maxlen=5)

        print(f"\nSistem hazır: SR={self.sr} Hz | "
              f"Pencere={SEGMENT_DURATION}s | Hop={HOP_DURATION}s")

    # ── Ses Callback (sounddevice thread) ────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[SES UYARI] {status}", file=sys.stderr)

        audio_chunk = indata[:, 0].copy()   # mono
        self.buffer.extend(audio_chunk)
        self.samples_since_last += frames

        if self.samples_since_last >= self.hop_len:
            if len(self.buffer) >= self.seg_len:
                segment = np.array(list(self.buffer)[-self.seg_len:])
                try:
                    self.audio_queue.put_nowait(segment)
                except queue.Full:
                    pass   # Yavaş işleme durumunda eski kareyi atla
            self.samples_since_last = 0

    # ── İşleme Thread'i ──────────────────────────────────────────────────────

    def _processing_loop(self):
        while self._running:
            try:
                segment = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t_start = time.perf_counter()
            pred_class, confidence, probs = predict(segment, self.model_pkg)
            t_elapsed = time.perf_counter() - t_start

            self.result_queue.put({
                "class":      pred_class,
                "confidence": confidence,
                "probs":      probs,
                "latency_ms": t_elapsed * 1000,
                "timestamp":  datetime.now().strftime("%H:%M:%S"),
            })

    # ── Çıktı Gösterimi ──────────────────────────────────────────────────────

    def _display_result(self, result: dict):
        cls  = result["class"]
        conf = result["confidence"]
        lat  = result["latency_ms"]
        ts   = result["timestamp"]
        probs = result["probs"]

        self.history.append(cls)

        # Basit çoğunluk oylaması (son 5 tahmin)
        if self.history:
            from collections import Counter
            majority = Counter(self.history).most_common(1)[0][0]
        else:
            majority = cls

        bar_len  = 30
        filled   = int(conf * bar_len)
        conf_bar = "█" * filled + "░" * (bar_len - filled)

        # Terminali temizle (gerçek zamanlı görünüm)
        print("\033[H\033[J", end="")   # ANSI temizleme

        print("╔══════════════════════════════════════════════════╗")
        print("║       IHA SES SINIFLANDIRMA SİSTEMİ             ║")
        print("╠══════════════════════════════════════════════════╣")
        print(f"║  Zaman     : {ts:<36} ║")
        print(f"║  Tespit    : {cls:<36} ║")
        print(f"║  Güven     : {conf*100:5.1f}%  [{conf_bar}] ║")
        print(f"║  Gecikme   : {lat:6.1f} ms{'':<29} ║")
        print(f"║  Oylama    : {majority:<36} ║")
        print("╠══════════════════════════════════════════════════╣")
        print(f"║  {'Sınıf':<20}  {'Olasılık':>10}  {'Bar':>14}  ║")
        print("║  " + "─" * 46 + "  ║")

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for name, prob in sorted_probs[:DISPLAY_TOP_N]:
            bar = "▌" * int(prob * 20)
            print(f"║  {name:<20}  {prob*100:9.1f}%  {bar:<14}  ║")

        print("╚══════════════════════════════════════════════════╝")
        print("  [Ctrl+C] ile durdurun")

    # ── Başlat / Durdur ──────────────────────────────────────────────────────

    def run(self):
        """Sistemi başlatır — Ctrl+C ile durur."""
        self._running = True

        self._proc_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._proc_thread.start()

        print(f"\nMikrofon dinleniyor... (cihaz: {'varsayılan' if self.device is None else self.device})")
        print("Ctrl+C ile durdurun.\n")

        try:
            with sd.InputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                blocksize=self.hop_len,
                device=self.device,
                callback=self._audio_callback,
            ):
                while self._running:
                    try:
                        result = self.result_queue.get(timeout=0.1)
                        self._display_result(result)
                    except queue.Empty:
                        pass

        except KeyboardInterrupt:
            print("\n\nSistem durduruldu.")
        finally:
            self._running = False
            if self._proc_thread:
                self._proc_thread.join(timeout=2.0)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def list_devices():
    print("\nMevcut ses cihazları:")
    print(sd.query_devices())
    print("\nGiriş cihazı indeksini --device parametresiyle belirtin.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IHA Ses Sınıflandırıcı — Gerçek Zamanlı")
    parser.add_argument("--model",        default=os.path.join(MODEL_DIR, "iha_model.joblib"),
                        help="Eğitilmiş model dosyası")
    parser.add_argument("--device", "-d", type=int, default=MIC_DEVICE,
                        help="Mikrofon cihaz indeksi (--list-devices ile görüntüle)")
    parser.add_argument("--list-devices", action="store_true",
                        help="Ses cihazlarını listele ve çık")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    model_pkg = load_model(args.model)
    classifier = RealTimeIHAClassifier(model_pkg, device=args.device)
    classifier.run()
