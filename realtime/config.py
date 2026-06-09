"""
İHA Ses Sınıflandırma — Realtime Demo Yapılandırması
Eski iha_sistemi/config.py'dan bağımsız, seg_duration=3.0 düzeltilmiş.
"""
import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR   = os.path.dirname(BASE_DIR)

# ─── Modeller ────────────────────────────────────────────────────────────────
_MODELS_DIR   = os.path.join(BASE_DIR, "models")
MODEL_KNN     = os.path.join(_MODELS_DIR, "model_knn.joblib")
MODEL_SVM     = os.path.join(_MODELS_DIR, "model_svm.joblib")
MODEL_RF      = os.path.join(_MODELS_DIR, "model_rf.joblib")

# Eski tek model (geriye dönük uyumluluk)
DEFAULT_MODEL = MODEL_KNN

# Tüm modeller sözlüğü
ALL_MODELS = {
    "kNN":  MODEL_KNN,
    "SVM":  MODEL_SVM,
    "RF":   MODEL_RF,
}

# ─── Ses İşleme ──────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000
SEGMENT_DURATION = 3.0    # Model bu süreyle eğitildi — değiştirme
HOP_DURATION     = 1.5    # Kaydırma adımı (pencere boyutunun yarısı)

# ─── Demo Ses ────────────────────────────────────────────────────────────────
DEMO_AUDIO_PATH = os.path.join(BASE_DIR, "demo_audio.wav")
DEMO_GT_PATH    = os.path.join(BASE_DIR, "demo_ground_truth.json")

# Her sınıftan kaç saniye alınacak
DEMO_BLOCK_DURATION = 30.0   # saniye
DEMO_SILENCE_DURATION = 2.0  # bloklar arası sessizlik

# Ham ses dosyaları
SARSILMAZ_DIR = os.path.join(PROJ_DIR, "SARSILMAZ İHA SESLER")
SARSILMAZ_FILES = {
    "BÜRKÜT":  os.path.join(SARSILMAZ_DIR, "BÜRKÜT",  "bürküt-flight.wav"),
    "DİNOZOR": os.path.join(SARSILMAZ_DIR, "DİNOZOR", "dino-flight-1.wav"),
    "HEXA":    os.path.join(SARSILMAZ_DIR, "HEXA",    "hexa-flight-1.wav"),
    "ÖRÜMCEK": os.path.join(SARSILMAZ_DIR, "ÖRÜMCEK", "örümcek-flight.WAV"),
}

# ─── Özellik Çıkarma ─────────────────────────────────────────────────────────
NUM_LEVELS = 7
WAVELET    = "sym4"
NUM_BINS   = 256

# ─── UI ──────────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "BÜRKÜT":  "#1f77b4",   # mavi
    "DİNOZOR": "#2ca02c",   # yeşil
    "HEXA":    "#ff7f0e",   # turuncu
    "ÖRÜMCEK": "#d62728",   # kırmızı
    "Belirsiz": "#7f7f7f",  # gri
}

CONFIDENCE_THRESHOLD = 0.40
HISTORY_WINDOW       = 30   # timeline'da gösterilecek son N tahmin
