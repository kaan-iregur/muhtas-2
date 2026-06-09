# Jetson Orin Nano Deployment Planı
### İHA Ses Sınıflandırma Sistemi — Gömülü Karta Geçiş

Hazırlayan: Claude (Kaan ile birlikte)  
Tarih: 2026-06-08

---

## Mevcut Durum (Başlangıç Noktası)

Sistem şu an PC'de çalışıyor:

```
Ses → DWT+TSK-LHP (NumPy, CPU) → 5376 özellik → sklearn kNN/SVM (CPU) → Tahmin
```

Jetson'a taşınacak dosyalar:

| Dosya | Görevi |
|-------|--------|
| `iha_sistemi/feature_extractor.py` | DWT+TSK-LHP özellik çıkarımı |
| `iha_sistemi/train.py` | Model eğitimi |
| `iha_sistemi/realtime.py` | Gerçek zamanlı mikrofon sistemi |
| `iha_sistemi/config.py` | Tüm parametreler |

---

## Neden Jetson'a Geçiyoruz?

- Sahada pil ile çalışmak için düşük güç tüketimi (5–15W)
- İHA'ya veya gözetleme sistemine montaj için küçük form factor
- Kameradan/radardan bağımsız, düşük maliyetli akustik tespit

---

## Seçilen Yaklaşım: Seçenek A (Minimal Değişim)

Özellik çıkarımına **dokunmuyoruz** — zaten hızlı (Jetson'da ~20–45ms, bütçe 500ms).  
Sadece **sınıflandırıcıyı** sklearn'den PyTorch MLP'ye çekiyoruz, TensorRT'ye dönüştürüyoruz.

### Neden tam CNN'e (Seçenek B) geçmiyoruz?
- Çok daha fazla veri gerektirir (elimizdeki veri yetersiz kalabilir)
- Tüm pipeline yeniden yazılır, risk yüksek
- Mevcut doğruluk (%87–94) zaten iyi — bozmaya gerek yok

---

## Adım Adım Yapılacaklar

### ADIM 1 — Jetson'a Ortam Kurulumu

Jetson Orin Nano üzerinde terminal açılır, sırayla:

```bash
# Python bağımlılıkları
pip install numpy pywavelets soundfile sounddevice joblib
pip install torch torchvision torchaudio  # Jetson'a özel wheel gerekebilir

# Jetson'a özel PyTorch: https://developer.nvidia.com/embedded/downloads
# JetPack versiyonuna göre doğru wheel indirilmeli

# TensorRT genellikle JetPack ile birlikte gelir
# Kontrol: python -c "import tensorrt; print(tensorrt.__version__)"
```

**Dikkat:** Normal `pip install torch` Jetson'da çalışmaz. NVIDIA'nın Jetson-özel wheel'ini indirmek gerekir. JetPack sürümüne göre URL değişiyor.

---

### ADIM 2 — Mevcut Modeli Test Et (Doğrulama)

Önce sklearn modelinin Jetson'da doğru çalıştığını doğrula:

```bash
# Modeli PC'den Jetson'a kopyala (scp veya USB)
scp iha_sistemi/models/iha_model.joblib jetson@<ip>:~/iha_sistemi/models/

# Jetson'da test et
python realtime.py --list-devices   # Mikrofon görünüyor mu?
python realtime.py                  # Çalışıyor mu?
```

Eğer bu adımda sistem çalışıyorsa **zaten kullanılabilir durumda** — gerisi optimizasyon.

---

### ADIM 3 — PyTorch MLP Sınıflandırıcısını Yaz

`train.py`'a eklenecek yeni bölüm:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class IHAClassifier(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    def forward(self, x):
        return self.net(x)
```

- Girdi: 35 özellik (mevcut özellik seçiminden geliyor, değişmiyor)
- Çıktı: N_sınıf (4 veya 5)
- Eğitim: PC'de PyTorch ile, ~50 epoch yeterli (küçük veri)

---

### ADIM 4 — ONNX'e Export

Eğitim bittikten sonra tek satır:

```python
dummy_input = torch.randn(1, 35)   # 35 özellik, 1 örnek
torch.onnx.export(
    model, dummy_input, "iha_model.onnx",
    input_names=["features"],
    output_names=["logits"],
    dynamic_axes={"features": {0: "batch"}}
)
```

Çıkan `iha_model.onnx` dosyası Jetson'a kopyalanır.

---

### ADIM 5 — TensorRT Engine Oluştur (Jetson üzerinde)

Jetson terminalinde:

```bash
trtexec --onnx=iha_model.onnx \
        --saveEngine=iha_model.engine \
        --fp16              # float16 ile ~2x hız artışı
```

Bu komut Jetson'a özel `.engine` dosyası üretir — başka cihazda çalışmaz.

---

### ADIM 6 — realtime.py'ı Güncelle

Tek değişen kısım `predict()` fonksiyonu:

**Şu an (sklearn):**
```python
pred_idx = clf.predict(X_sc)[0]
```

**Yeni (TensorRT):**
```python
import tensorrt as trt
import pycuda.driver as cuda

# Başlangıçta bir kez yükle
engine = load_engine("iha_model.engine")
context = engine.create_execution_context()

# Her tahmin için
input_tensor = torch.from_numpy(X_sc).float().cuda()
context.execute_v2(bindings)
output = output_tensor.cpu().numpy()
pred_idx = np.argmax(output)
```

`feature_extractor.py` **hiç değişmiyor.**

---

### ADIM 7 — Gecikme Testi

Sistem çalışırken terminalde görünen `Gecikme: X ms` değeri izlenir.

Beklenen sonuçlar:

| Bileşen | Jetson sklearn | Jetson TensorRT |
|---------|---------------|----------------|
| `extract_features()` | ~20–45ms | ~20–45ms (değişmez) |
| Sınıflandırıcı | ~3–5ms | **~0.1–0.5ms** |
| **Toplam** | ~25–50ms | **~21–46ms** |
| Mevcut bütçe | 500ms | 500ms |

Fark küçük görünür çünkü darboğaz zaten feature extraction. Asıl kazanç **güç tüketimi ve GPU'nun aktif kullanılması**.

---

### ADIM 8 (İleride) — Background Sınıfı Ekle

**Kritik eksik:** Sistem şu an her zaman bir İHA ismi söylüyor.  
Mikrofona el çırpsan bile "DJI Air2S" diyebilir.

Yapılacak:
- Sessizlik / arka plan ses kayıtları toplanır
- Yeni bir sınıf olarak eklenir: `"background"` veya `"sessiz"`
- Model yeniden eğitilir
- `confidence < 0.5` kontrolü hâlâ çalışıyor ama gerçek background sınıfı çok daha güvenilir

---

## Özet Kontrol Listesi

- [ ] Jetson'da JetPack sürümü öğrenilir
- [ ] PyTorch Jetson wheel'i indirilir ve kurulur
- [ ] `iha_model.joblib` Jetson'a kopyalanır, sklearn ile test edilir
- [ ] PC'de PyTorch MLP yazılır ve eğitilir
- [ ] `iha_model.onnx` export edilir
- [ ] Jetson'da `trtexec` ile `.engine` oluşturulur
- [ ] `realtime.py`'daki `predict()` TensorRT'ye güncellenir
- [ ] Jetson'da `realtime.py` çalıştırılır, gecikme ölçülür
- [ ] (Sonraki aşama) Background sınıfı eklenir

---

## Notlar

- Eğitim her zaman PC'de yapılır, Jetson'a sadece model kopyalanır
- `.engine` dosyası Jetson'a özel — başka cihazda çalışmaz, her Jetson için yeniden oluşturulur
- JetPack 5.x ve 6.x arasında TensorRT API değişikliği var, sürüme dikkat
- `feature_extractor.py` hiç değişmediği için mevcut test sonuçları geçerliliğini koruyor
