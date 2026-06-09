"""
Unified Test Pipeline — MDWT + TSK-LHP
=======================================
3 veri seti, tutarlı metodoloji:

  Dataset 1: Sentetik  — 5 sınıf (DJI_Air2S, DJI_Matrice, DJI_Mavic3,
                                    Parrot_Anafi, Sabit_Kanat)
  Dataset 2: Sarsilmaz — 4 sınıf (BÜRKÜT, DİNOZOR, HEXA, ÖRÜMCEK)
  Dataset 3: Saraalemadi — 3 sınıf (bebop_1, membo_1, unknown)

Her dataset için:
  [A] 10-Fold Stratified CV  → kNN, SVM, RF
  [B] Bağımsız Split          → kNN, SVM, RF
       - Sentetik   : dataset_train → dataset_test (ayrı seed, en dürüst)
       - Sarsilmaz  : temporal split (ilk %70 eğitim / son %30 test)
       - Saraalemadi: %80/%20 stratified random split

Kullanım:
  python unifiye_test.py --dataset all
  python unifiye_test.py --dataset sentetik
  python unifiye_test.py --dataset sarsilmaz
  python unifiye_test.py --dataset saraalemadi
"""

import os, sys, glob, time, argparse, warnings, zipfile, urllib.request
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix)
from sklearn.feature_selection import mutual_info_classif, chi2

warnings.filterwarnings("ignore")

# ── Yollar ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.resolve()
PROJ      = ROOT.parent

SYS_PATH  = str(PROJ / "iha_sistemi")
sys.path.insert(0, SYS_PATH)
from feature_extractor import extract_features, load_audio
from config import SAMPLE_RATE

SENTETIK_TRAIN  = PROJ / "test" / "dataset_train"
SENTETIK_TEST   = PROJ / "test" / "dataset_test"
SARS_CACHE_TR   = PROJ / "test-2" / "cache_v3_train.npz"
SARS_CACHE_TE   = PROJ / "test-2" / "cache_v3_test.npz"
SARA_URL        = "https://github.com/saraalemadi/DroneAudioDataset/archive/refs/heads/master.zip"
SARA_ZIP        = ROOT / "saraalemadi_tmp.zip"
SARA_DIR        = ROOT / "saraalemadi_tmp"

OUT_SENTETIK    = ROOT / "01_sentetik_5sinif"  / "sonuclar"
OUT_SARSILMAZ   = ROOT / "02_sarsilmaz_4sinif"  / "sonuclar"
OUT_SARAALEMADI = ROOT / "03_saraalemadi_3sinif" / "sonuclar"

# ── Sabitler ──────────────────────────────────────────────────────────────────
NUM_LEVELS    = 7
NUM_BINS      = 256
FEATURES_PER  = 5
SELECTION_RUNS = 5
KNN_K         = 5
CV_FOLDS      = 10
SARA_MAX_UNK  = 700


# ══════════════════════════════════════════════════════════════════════════════
#  ORTAK YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def level_slices():
    n = NUM_BINS * 3
    return [(i * n, (i + 1) * n) for i in range(NUM_LEVELS)]

def select_mi(X, y, k=FEATURES_PER, seed=0):
    sel = []
    for s, e in level_slices():
        sc = mutual_info_classif(X[:, s:e], y, random_state=seed)
        sel.extend((s + np.argsort(sc)[::-1][:k]).tolist())
    return np.array(sel, np.int32)

def select_chi2(X, y, k=FEATURES_PER):
    sel = []
    for s, e in level_slices():
        sc, _ = chi2(X[:, s:e] + 1e-10, y)
        sel.extend((s + np.argsort(sc)[::-1][:k]).tolist())
    return np.array(sel, np.int32)

def best_feature_set(X_tr, y_tr):
    """10 aday özellik setinden (5×MI + 5×Chi2) CV'de en iyi indeksi seç."""
    best_acc, best_idx = 0.0, None
    clf_ref = KNeighborsClassifier(KNN_K, metric="cityblock")
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42)
    for run in range(SELECTION_RUNS):
        for fn in (lambda: select_mi(X_tr, y_tr, seed=run * 7),
                   lambda: select_chi2(X_tr, y_tr)):
            idx = fn()
            Xs  = StandardScaler().fit_transform(X_tr[:, idx])
            acc = cross_val_score(clf_ref, Xs, y_tr, cv=skf,
                                  scoring="accuracy", n_jobs=-1).mean()
            if acc > best_acc:
                best_acc, best_idx = acc, idx
    return best_idx

def make_classifiers():
    return [
        ("kNN", KNeighborsClassifier(KNN_K, metric="cityblock")),
        ("SVM", SVC(kernel="poly", degree=3, C=1.0, coef0=1.0,
                    probability=True, random_state=42)),
        ("RF",  RandomForestClassifier(n_estimators=200, random_state=42,
                                        n_jobs=-1)),
    ]

def run_cv(X, y, idx):
    """10-fold CV sonuçları: [(name, acc, f1), ...]"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X[:, idx])
    skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=42)
    out = []
    for name, clf in make_classifiers():
        accs = cross_val_score(clf, Xs, y, cv=skf,
                               scoring="accuracy", n_jobs=-1)
        f1s  = cross_val_score(clf, Xs, y, cv=skf,
                               scoring="f1_weighted", n_jobs=-1)
        # Sınıf bazlı: tam veri üzerinde fold tahminleri birleştir
        all_pred, all_true = [], []
        for tr_i, te_i in skf.split(Xs, y):
            clf.fit(Xs[tr_i], y[tr_i])
            all_pred.extend(clf.predict(Xs[te_i]))
            all_true.extend(y[te_i])
        out.append({"name": name,
                    "acc": float(accs.mean()),
                    "acc_std": float(accs.std()),
                    "f1":  float(f1s.mean()),
                    "pred": np.array(all_pred),
                    "true": np.array(all_true)})
    return out, scaler

def run_split(X_tr, y_tr, X_te, y_te, idx, scaler_fit_on_tr=True):
    """Bağımsız split sonuçları."""
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_tr[:, idx])
    Xs_te = scaler.transform(X_te[:, idx])
    out = []
    for name, clf in make_classifiers():
        clf.fit(Xs_tr, y_tr)
        pred = clf.predict(Xs_te)
        out.append({"name": name,
                    "acc": accuracy_score(y_te, pred),
                    "f1":  f1_score(y_te, pred, average="weighted"),
                    "pred": pred,
                    "true": y_te})
    return out


def write_report(title, split_label, classes, cv_results, split_results,
                 meta, out_dir: Path):
    """Tutarlı metin raporu yaz."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def cm_str(pred, true, classes):
        cm = confusion_matrix(true, pred)
        lines = ["  " + "  ".join(f"{c:>10s}" for c in classes)]
        for i, row in enumerate(cm):
            lines.append(f"{classes[i]:>10s}  " +
                         "  ".join(f"{v:>10d}" for v in row))
        return "\n".join(lines)

    lines = []
    lines.append("╔" + "═"*66 + "╗")
    lines.append(f"║  {title:<64}║")
    lines.append("╚" + "═"*66 + "╝")
    lines.append("")
    for k, v in meta.items():
        lines.append(f"  {k:<25s}: {v}")
    lines.append("")

    # ── 10-Fold CV ────────────────────────────────────────────────────────────
    lines.append("10-FOLD STRATIFIED CV")
    lines.append("─" * 68)
    lines.append(f"  {'Model':<8}  {'Doğruluk':>10}  {'± Std':>8}  {'F1 (ağır.)':>12}")
    lines.append(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*12}")
    for r in cv_results:
        lines.append(f"  {r['name']:<8}  {r['acc']*100:>9.2f}%"
                     f"  {r['acc_std']*100:>7.2f}%  {r['f1']*100:>11.2f}%")
    lines.append("")
    best_cv = max(cv_results, key=lambda r: r["acc"])
    lines.append(f"  En iyi CV: {best_cv['name']} → %{best_cv['acc']*100:.2f}")
    lines.append("")
    lines.append(f"  Sınıflandırma Raporu ({best_cv['name']}, 10-fold pred birleştirme):")
    lines.append(classification_report(best_cv["true"], best_cv["pred"],
                                        target_names=classes,
                                        digits=3))
    lines.append("  Karmaşıklık Matrisi:")
    lines.append(cm_str(best_cv["pred"], best_cv["true"], classes))
    lines.append("")

    # ── Bağımsız Split ────────────────────────────────────────────────────────
    lines.append(f"BAĞIMSIZ SPLIT ({split_label})")
    lines.append("─" * 68)
    lines.append(f"  {'Model':<8}  {'Doğruluk':>10}  {'F1 (ağır.)':>12}")
    lines.append(f"  {'─'*8}  {'─'*10}  {'─'*12}")
    for r in split_results:
        lines.append(f"  {r['name']:<8}  {r['acc']*100:>9.2f}%  {r['f1']*100:>11.2f}%")
    lines.append("")
    best_sp = max(split_results, key=lambda r: r["acc"])
    lines.append(f"  En iyi Split: {best_sp['name']} → %{best_sp['acc']*100:.2f}")
    lines.append("")
    lines.append(f"  Sınıflandırma Raporu ({best_sp['name']}):")
    lines.append(classification_report(best_sp["true"], best_sp["pred"],
                                        target_names=classes,
                                        digits=3))
    lines.append("  Karmaşıklık Matrisi:")
    lines.append(cm_str(best_sp["pred"], best_sp["true"], classes))
    lines.append("")
    lines.append(f"Tarih: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    report = "\n".join(lines)
    rpath = out_dir / "rapor.txt"
    rpath.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n  → Rapor kaydedildi: {rpath}")
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — SENTETİK
# ══════════════════════════════════════════════════════════════════════════════

def load_folder(folder: Path, seg_dur=1.0) -> tuple[np.ndarray, np.ndarray, list]:
    """Klasör altındaki tüm ses dosyalarını yükle → (X, y, classes)."""
    classes = sorted([d.name for d in folder.iterdir() if d.is_dir()])
    X_list, y_list = [], []
    seg_len = int(SAMPLE_RATE * seg_dur)
    for ci, cls in enumerate(classes):
        files = sorted(glob.glob(str(folder / cls / "*")))
        for fpath in files:
            try:
                audio, sr = load_audio(fpath, SAMPLE_RATE)
                for start in range(0, len(audio) - seg_len + 1, seg_len):
                    feat = extract_features(audio[start:start + seg_len])
                    if feat is not None:
                        X_list.append(feat)
                        y_list.append(ci)
            except Exception as e:
                print(f"    HATA ({Path(fpath).name}): {e}")
    return np.array(X_list, np.float32), np.array(y_list, np.int32), classes


def run_sentetik():
    print("\n" + "="*68)
    print("  DATASET 1 — SENTETİK (5 Sınıf)")
    print("="*68)

    # Önbellek kontrol
    cache_tr = ROOT / "01_sentetik_5sinif" / "cache_train.npz"
    cache_te = ROOT / "01_sentetik_5sinif" / "cache_test.npz"

    if cache_tr.exists() and cache_te.exists():
        print("  Önbellek bulundu, yükleniyor...")
        d = np.load(cache_tr, allow_pickle=True)
        X_tr, y_tr, classes = d["X"], d["y"], list(d["classes"])
        d = np.load(cache_te, allow_pickle=True)
        X_te, y_te = d["X"], d["y"]
    else:
        print("  ADIM 1 — Eğitim verisi özellik çıkarımı...")
        X_tr, y_tr, classes = load_folder(SENTETIK_TRAIN, seg_dur=3.0)
        print(f"    Train: {X_tr.shape[0]} segment, {len(classes)} sınıf")
        np.savez_compressed(cache_tr, X=X_tr, y=y_tr, classes=np.array(classes))

        print("  ADIM 2 — Test verisi özellik çıkarımı...")
        X_te, y_te, _ = load_folder(SENTETIK_TEST, seg_dur=3.0)
        print(f"    Test: {X_te.shape[0]} segment")
        np.savez_compressed(cache_te, X=X_te, y=y_te, classes=np.array(classes))

    print(f"\n  Sınıflar: {classes}")
    for i, c in enumerate(classes):
        print(f"    {c}: train={( y_tr==i).sum()}  test={(y_te==i).sum()}")

    # Özellik seçimi: train üzerinde en iyi seti bul
    print("\n  ADIM 3 — Özellik seçimi (train verisi ile)...")
    best_idx = best_feature_set(X_tr, y_tr)
    print(f"    Seçilen özellik sayısı: {len(best_idx)}")

    # 10-Fold CV (sadece train verisi üzerinde)
    print("\n  ADIM 4 — 10-Fold CV (train verisi)...")
    cv_res, _ = run_cv(X_tr, y_tr, best_idx)
    for r in cv_res:
        print(f"    {r['name']}: CV={r['acc']*100:.2f}% ± {r['acc_std']*100:.2f}%  F1={r['f1']*100:.2f}%")

    # Bağımsız split (train → test, farklı seed, en dürüst)
    print("\n  ADIM 5 — Bağımsız Split (train → test, farklı sentetik seed)...")
    sp_res = run_split(X_tr, y_tr, X_te, y_te, best_idx)
    for r in sp_res:
        print(f"    {r['name']}: Test doğruluk={r['acc']*100:.2f}%  F1={r['f1']*100:.2f}%")

    meta = {
        "Veri tipi"        : "Sentetik (sinüzoidal İHA ses modeli)",
        "Sınıf sayısı"     : "5",
        "Sınıflar"         : ", ".join(classes),
        "Train segment"    : str(X_tr.shape[0]),
        "Test segment"     : str(X_te.shape[0]),
        "Segment süresi"   : "3.0s",
        "Örnekleme"        : "16000 Hz",
        "Ham özellik"      : "5376 (7×768)",
        "Seçilen özellik"  : str(len(best_idx)),
        "Split türü"       : "BAĞIMSIZ SEED (train/test farklı rastgele tohum)",
    }
    write_report("SENTETİK VERİ — 5 Sınıf İHA Ses Tanıma",
                 "Bağımsız Seed (en dürüst yöntem)",
                 classes, cv_res, sp_res, meta, OUT_SENTETIK)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — SARSİLMAZ
# ══════════════════════════════════════════════════════════════════════════════

def run_sarsilmaz():
    print("\n" + "="*68)
    print("  DATASET 2 — SARSİLMAZ GERÇEK VERİ (4 Sınıf)")
    print("="*68)

    # Cache hazır — yükle
    d_tr = np.load(SARS_CACHE_TR, allow_pickle=True)
    d_te = np.load(SARS_CACHE_TE, allow_pickle=True)
    X_tr, y_tr = d_tr["X"].astype(np.float32), d_tr["y"].astype(np.int32)
    X_te, y_te = d_te["X"].astype(np.float32), d_te["y"].astype(np.int32)
    classes    = list(d_tr["classes"])

    print(f"  Sınıflar: {classes}")
    for i, c in enumerate(classes):
        print(f"    {c}: train={(y_tr==i).sum()}  test={(y_te==i).sum()}")

    # Özellik seçimi
    print("\n  ADIM 1 — Özellik seçimi (train verisi ile)...")
    best_idx = best_feature_set(X_tr, y_tr)
    print(f"    Seçilen özellik sayısı: {len(best_idx)}")

    # 10-Fold CV (tüm veri: train + test birleştirilerek)
    X_all = np.vstack([X_tr, X_te])
    y_all = np.concatenate([y_tr, y_te])
    print("\n  ADIM 2 — 10-Fold CV (tüm veri)...")
    cv_res, _ = run_cv(X_all, y_all, best_idx)
    for r in cv_res:
        print(f"    {r['name']}: CV={r['acc']*100:.2f}% ± {r['acc_std']*100:.2f}%  F1={r['f1']*100:.2f}%")

    # Temporal split
    print("\n  ADIM 3 — Temporal Split (ilk %70 → son %30)...")
    sp_res = run_split(X_tr, y_tr, X_te, y_te, best_idx)
    for r in sp_res:
        print(f"    {r['name']}: Test doğruluk={r['acc']*100:.2f}%  F1={r['f1']*100:.2f}%")

    meta = {
        "Veri tipi"        : "Gerçek Sarsilmaz saha kaydı",
        "Sınıf sayısı"     : "4",
        "Sınıflar"         : ", ".join(classes),
        "Train segment"    : str(X_tr.shape[0]),
        "Test segment"     : str(X_te.shape[0]),
        "Segment süresi"   : "3.0s (augmentasyon uygulandı)",
        "Örnekleme"        : "16000 Hz (48kHz'den resample)",
        "Ham özellik"      : "5376 (7×768)",
        "Seçilen özellik"  : str(len(best_idx)),
        "Split türü"       : "TEMPORAL (ilk %70 eğitim / son %30 test)",
    }
    write_report("SARSİLMAZ GERÇEK VERİ — 4 Sınıf İHA Ses Tanıma",
                 "Temporal Split (kronolojik)",
                 classes, cv_res, sp_res, meta, OUT_SARSILMAZ)


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — SARAALEMADİ
# ══════════════════════════════════════════════════════════════════════════════

def download_saraalemadi():
    print("  Veri seti indiriliyor (~550 MB)...")
    SARA_DIR.mkdir(exist_ok=True)
    urllib.request.urlretrieve(SARA_URL, SARA_ZIP)
    print("  ZIP açılıyor...")
    with zipfile.ZipFile(SARA_ZIP, "r") as z:
        members = [m for m in z.namelist()
                   if "Multiclass_Drone_Audio" in m]
        z.extractall(SARA_DIR, members=members)
    SARA_ZIP.unlink()
    print("  Tamamlandı.")

def load_saraalemadi():
    import soundfile as sf

    multiclass = SARA_DIR / "DroneAudioDataset-master" / "Multiclass_Drone_Audio"
    if not multiclass.exists():
        download_saraalemadi()

    CLASS_MAP = {"bebop_1": 0, "membo_1": 1, "unknown": 2}
    classes   = ["bebop_1", "membo_1", "unknown"]
    X_list, y_list = [], []
    seg_len = int(SAMPLE_RATE * 1.0)

    for cls_name, label in CLASS_MAP.items():
        cls_dir = multiclass / cls_name
        files   = sorted([f for f in os.listdir(cls_dir)
                          if f.lower().endswith(".wav")])
        # unknown dengele
        if cls_name == "unknown" and len(files) > SARA_MAX_UNK:
            step  = len(files) // SARA_MAX_UNK
            files = files[::step][:SARA_MAX_UNK]

        print(f"    [{cls_name}] {len(files)} dosya...", flush=True)
        for fname in files:
            fpath = cls_dir / fname
            try:
                audio, fsr = sf.read(str(fpath), dtype="float32", always_2d=False)
                if audio.ndim > 1: audio = audio.mean(axis=1)
                if fsr != SAMPLE_RATE:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=fsr, target_sr=SAMPLE_RATE)
                if len(audio) < int(SAMPLE_RATE * 0.8): continue
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                for start in range(0, len(audio) - seg_len + 1, seg_len):
                    feat = extract_features(audio[start:start + seg_len])
                    if feat is not None:
                        X_list.append(feat)
                        y_list.append(label)
            except Exception as e:
                pass

    return np.array(X_list, np.float32), np.array(y_list, np.int32), classes


def run_saraalemadi():
    print("\n" + "="*68)
    print("  DATASET 3 — SARAAlEMADİ PUBLIC (3 Sınıf)")
    print("="*68)

    cache_path = ROOT / "03_saraalemadi_3sinif" / "cache.npz"

    if cache_path.exists():
        print("  Önbellek bulundu, yükleniyor...")
        d = np.load(cache_path, allow_pickle=True)
        X, y, classes = d["X"], d["y"], list(d["classes"])
    else:
        print("  ADIM 1 — Veri hazırlama...")
        X, y, classes = load_saraalemadi()
        np.savez_compressed(cache_path, X=X, y=y, classes=np.array(classes))
        # İndirilen ham veriyi temizle
        import shutil
        if SARA_DIR.exists():
            shutil.rmtree(SARA_DIR)
            print("  Geçici veri silindi.")

    print(f"  Sınıflar: {classes}")
    for i, c in enumerate(classes):
        print(f"    {c}: {(y==i).sum()} segment")

    # Özellik seçimi (tüm veri üzerinde)
    print("\n  ADIM 2 — Özellik seçimi...")
    best_idx = best_feature_set(X, y)
    print(f"    Seçilen özellik sayısı: {len(best_idx)}")

    # 10-Fold CV
    print("\n  ADIM 3 — 10-Fold CV...")
    cv_res, _ = run_cv(X, y, best_idx)
    for r in cv_res:
        print(f"    {r['name']}: CV={r['acc']*100:.2f}% ± {r['acc_std']*100:.2f}%  F1={r['f1']*100:.2f}%")

    # %80/%20 Stratified Random Split
    print("\n  ADIM 4 — %80/%20 Stratified Random Split...")
    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X[:, best_idx])
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        Xs, y, test_size=0.20, random_state=42, stratify=y)

    sp_res = []
    for name, clf in make_classifiers():
        clf.fit(X_tr_s, y_tr_s)
        pred = clf.predict(X_te_s)
        sp_res.append({"name": name,
                       "acc": accuracy_score(y_te_s, pred),
                       "f1":  f1_score(y_te_s, pred, average="weighted"),
                       "pred": pred, "true": y_te_s})
        print(f"    {name}: Test doğruluk={sp_res[-1]['acc']*100:.2f}%  F1={sp_res[-1]['f1']*100:.2f}%")

    meta = {
        "Veri tipi"        : "Public drone ses (github.com/saraalemadi/DroneAudioDataset)",
        "Sınıf sayısı"     : "3",
        "Sınıflar"         : ", ".join(classes),
        "Toplam segment"   : str(len(y)),
        "Segment süresi"   : "1.0s",
        "Örnekleme"        : "16000 Hz (native)",
        "Ham özellik"      : "5376 (7×768)",
        "Seçilen özellik"  : str(len(best_idx)),
        "Split türü"       : "RANDOM %80/%20 Stratified",
    }
    write_report("SARAAlEMADİ PUBLIC VERİ — 3 Sınıf Drone Ses Tanıma",
                 "%80/%20 Stratified Random",
                 classes, cv_res, sp_res, meta, OUT_SARAALEMADI)


# ══════════════════════════════════════════════════════════════════════════════
#  ANA PROGRAM
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all",
                        choices=["all", "sentetik", "sarsilmaz", "saraalemadi"])
    args = parser.parse_args()

    t0 = time.time()
    print("╔" + "═"*66 + "╗")
    print("║  MDWT+TSK-LHP — Unified Test Pipeline (3 Dataset)              ║")
    print("╚" + "═"*66 + "╝")

    if args.dataset in ("all", "sentetik"):
        run_sentetik()
    if args.dataset in ("all", "sarsilmaz"):
        run_sarsilmaz()
    if args.dataset in ("all", "saraalemadi"):
        run_saraalemadi()

    elapsed = time.time() - t0
    print(f"\n{'='*68}")
    print(f"  TÜMÜ TAMAMLANDI — Toplam süre: {elapsed/60:.1f} dakika")
    print(f"{'='*68}")
