"""
IHA Ses Sınıflandırma Sistemi - Yapılandırma
"""
import os

# Proje kök dizini (bu dosyanın bulunduğu yer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Veri Seti ───────────────────────────────────────────────────────────────
# Dataset klasörü altında her alt klasör bir IHA tipidir.
# Örn: dataset/DJI_Phantom/, dataset/DJI_Mavic/, dataset/Bayraktar/
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
EXCEL_DIR   = os.path.join(BASE_DIR, "excel_ciktilari")

# ─── Ses İşleme ──────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16000   # Hz — Jetson Nano için dengeli değer
SEGMENT_DURATION = 1.0     # saniye — eğitim ve çıkarım pencere uzunluğu
HOP_DURATION     = 0.5     # saniye — gerçek zamanlı ilerleme adımı

# ─── Özellik Çıkarma (MDWT + TSK-LHP) ───────────────────────────────────────
NUM_LEVELS = 7             # Dalgacık ayrıştırma seviyesi
WAVELET    = "sym4"        # Symlet-4 dalgacığı
NUM_BINS   = 256           # Histogram çözünürlüğü (0-255 = 8-bit)

# Her seviye başına özellik vektörü uzunluğu: NUM_BINS * 3 = 768
# Toplam özellik boyutu: NUM_LEVELS * 768 = 5376

# ─── Özellik Seçimi ──────────────────────────────────────────────────────────
FEATURES_PER_LEVEL  = 5    # Seviye başına seçilen özellik sayısı
SELECTION_RUNS      = 5    # Her seçim yöntemi kaç kez tekrarlanır
# Her çalışmada seçilen toplam: FEATURES_PER_LEVEL * NUM_LEVELS = 35

# ─── Sınıflandırıcılar ───────────────────────────────────────────────────────
KNN_K      = 5
KNN_METRIC = "cityblock"   # Manhattan mesafesi
SVM_KERNEL = "poly"
SVM_DEGREE = 3             # Kübik kernel
SVM_C      = 1.0
CV_FOLDS   = 10

# ─── Gerçek Zamanlı Sistem ───────────────────────────────────────────────────
MIC_DEVICE     = None      # None = sistem varsayılan mikrofonu
# Mevcut cihazları görmek için: python -c "import sounddevice as sd; print(sd.query_devices())"

CONFIDENCE_THRESHOLD = 0.5  # Bu güven altındaysa "Belirsiz" yazar
DISPLAY_TOP_N        = 3    # Kaç sınıfın skorunu göster
