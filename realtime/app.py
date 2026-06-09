"""
İHA Ses Sınıflandırma — Çok Model Gerçek Zamanlı Demo
======================================================
Kullanım:
  cd realtime/
  streamlit run app.py
"""

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    SAMPLE_RATE, SEGMENT_DURATION, HOP_DURATION,
    DEMO_AUDIO_PATH, DEMO_GT_PATH,
    ALL_MODELS, CLASS_COLORS, HISTORY_WINDOW, CONFIDENCE_THRESHOLD,
)
from model_adapter import load_model, predict as model_predict

# ─── Sayfa ayarı ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="İHA Ses Sınıflandırma",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .class-box {
    border-radius: 10px; padding: 14px 10px; text-align: center;
    font-size: 1.4rem; font-weight: 800; letter-spacing: 1px;
    color: white; text-shadow: 0 1px 4px rgba(0,0,0,0.5);
    margin-bottom: 6px;
  }
  .model-header {
    text-align: center; font-size: 1.1rem; font-weight: 700;
    color: #cdd6f4; margin-bottom: 8px; letter-spacing: 1px;
  }
  .match-ok  { color: #00cc66; font-weight: 700; }
  .match-err { color: #ff4444; font-weight: 700; }
  .stat-row  { background:#1e1e2e; border-radius:8px; padding:10px 14px; margin-bottom:6px; }
  .stat-name { color:#a6adc8; font-size:0.8rem; }
  .stat-val  { color:#cdd6f4; font-size:1.3rem; font-weight:800; }
  hr { border-color: #313244; }
</style>
""", unsafe_allow_html=True)

MODEL_LABEL_COLORS = {"kNN": "#89b4fa", "SVM": "#a6e3a1", "RF": "#fab387"}


# ─── Session state ────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "all_results":  [],
        "current_idx":  0,
        "playing":      False,
        "processed":    False,
        "models":       {},
        "speed":        1.0,
        "stats":        {},   # {model_name: {correct, total}}
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def get_gt_class(t: float, gt_blocks: list) -> str:
    for b in gt_blocks:
        if b["start_s"] <= t < b["end_s"]:
            return b["class"]
    return "?"


@st.cache_resource(show_spinner="Modeller yükleniyor...")
def load_all_models(model_paths: tuple) -> dict:
    result = {}
    for name, path in model_paths:
        result[name] = load_model(path)
    return result


def process_audio(audio_path: str, models: dict, gt_blocks: list) -> list:
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except ImportError:
            new_len = int(len(audio) * SAMPLE_RATE / sr)
            audio = np.interp(np.linspace(0, len(audio)-1, new_len),
                              np.arange(len(audio)), audio).astype(np.float32)

    seg_len = int(SEGMENT_DURATION * SAMPLE_RATE)
    hop_len = int(HOP_DURATION    * SAMPLE_RATE)
    results = []

    for start in range(0, len(audio) - seg_len + 1, hop_len):
        seg = audio[start: start + seg_len].astype(np.float64)
        t   = start / SAMPLE_RATE
        gt  = get_gt_class(t + SEGMENT_DURATION / 2, gt_blocks)

        preds = {}
        for name, m_info in models.items():
            cls, conf, probs = model_predict(seg, m_info)
            preds[name] = {
                "class":      cls,
                "confidence": conf,
                "probs":      probs,
                "match":      cls == gt,
            }

        results.append({
            "time":     t,
            "time_end": t + SEGMENT_DURATION,
            "gt":       gt,
            "preds":    preds,
        })

    return results


# ─── Grafikler ────────────────────────────────────────────────────────────────

def make_timeline(results: list, cidx: int) -> go.Figure:
    n    = min(cidx + 1, len(results))
    tail = results[max(0, n - HISTORY_WINDOW): n]

    times      = [r["time"] for r in tail]
    gt_labels  = [r["gt"]   for r in tail]
    model_names = list(ALL_MODELS.keys())

    rows  = ["GT"] + model_names
    n_row = len(rows)
    fig   = make_subplots(rows=1, cols=1)
    fig   = go.Figure()

    y_pos    = {name: i for i, name in enumerate(reversed(rows))}
    gt_color = [CLASS_COLORS.get(g, "#7f7f7f") for g in gt_labels]

    # Ground truth
    fig.add_trace(go.Scatter(
        x=times, y=[y_pos["GT"]] * len(times),
        mode="markers",
        marker=dict(color=gt_color, size=20, symbol="square",
                    line=dict(color="#ffffff", width=1)),
        text=[f"GT: {g}<br>t={t:.1f}s" for g, t in zip(gt_labels, times)],
        hovertemplate="%{text}<extra></extra>",
        name="Ground Truth",
    ))

    # Her model için satır
    for m_name in model_names:
        labels  = [r["preds"][m_name]["class"]      for r in tail]
        confs   = [r["preds"][m_name]["confidence"] for r in tail]
        matches = [r["preds"][m_name]["match"]       for r in tail]

        colors  = [CLASS_COLORS.get(l, "#7f7f7f") for l in labels]
        borders = ["#00ff88" if m else "#ff4444"   for m in matches]

        fig.add_trace(go.Scatter(
            x=times, y=[y_pos[m_name]] * len(times),
            mode="markers",
            marker=dict(color=colors, size=20, symbol="square",
                        line=dict(color=borders, width=3)),
            text=[f"{m_name}: {l}<br>%{c*100:.0f}<br>t={t:.1f}s"
                  for l, c, t in zip(labels, confs, times)],
            hovertemplate="%{text}<extra></extra>",
            name=m_name,
        ))

    # Sınıf renk lejantı
    for cls, col in CLASS_COLORS.items():
        if cls == "Belirsiz":
            continue
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=col, size=12, symbol="square"),
            name=cls,
        ))

    tick_labels = list(reversed(rows))
    fig.update_layout(
        height=270,
        margin=dict(l=5, r=5, t=10, b=30),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#cdd6f4", size=11),
        xaxis=dict(title="Zaman (s)", gridcolor="#2a2a3e", color="#a6adc8"),
        yaxis=dict(
            tickvals=list(range(n_row)),
            ticktext=tick_labels,
            gridcolor="#2a2a3e", color="#a6adc8",
            tickfont=dict(size=12),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def make_acc_bar(stats: dict) -> go.Figure:
    names = list(stats.keys())
    accs  = [
        (stats[n]["correct"] / stats[n]["total"] * 100)
        if stats[n]["total"] > 0 else 0
        for n in names
    ]
    colors = [MODEL_LABEL_COLORS.get(n, "#cdd6f4") for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=accs,
        marker_color=colors,
        text=[f"%{a:.1f}" for a in accs],
        textposition="outside",
        textfont=dict(size=14, color="#cdd6f4"),
    ))
    fig.update_layout(
        height=200, margin=dict(l=5, r=5, t=20, b=20),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#cdd6f4"),
        yaxis=dict(range=[0, 110], gridcolor="#2a2a3e",
                   title="Doğruluk (%)", color="#a6adc8"),
        xaxis=dict(color="#a6adc8"),
        showlegend=False,
    )
    return fig


# ─── Ana Uygulama ─────────────────────────────────────────────────────────────
st.markdown("## 🎯 İHA Ses Sınıflandırma — Çok Model Demo")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Kontroller")

    if not (os.path.exists(DEMO_AUDIO_PATH) and os.path.exists(DEMO_GT_PATH)):
        st.error("Demo ses dosyası bulunamadı.")
        st.code("python create_demo_audio.py", language="bash")
        st.stop()

    missing = [n for n, p in ALL_MODELS.items() if not os.path.exists(p)]
    if missing:
        st.error(f"Eksik model: {missing}")
        st.code("python train_all.py", language="bash")
        st.stop()

    # Modelleri yükle
    models = load_all_models(tuple(ALL_MODELS.items()))
    st.session_state.models = models

    for name, m in models.items():
        col = MODEL_LABEL_COLORS.get(name, "#cdd6f4")
        st.markdown(
            f'<span style="color:{col};font-weight:700;">{name}</span>  '
            f'CV=%{m.get("cv_accuracy", m["accuracy"])*100:.1f}  '
            f'Split=%{m["accuracy"]*100:.1f}',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if not st.session_state.processed:
        if st.button("📂 Ses Dosyasını İşle", type="primary",
                     use_container_width=True):
            gt_data = json.load(open(DEMO_GT_PATH, encoding="utf-8"))
            with st.spinner("İşleniyor... (3 model × 83 pencere)"):
                results = process_audio(DEMO_AUDIO_PATH, models,
                                        gt_data["blocks"])
            st.session_state.all_results = results
            st.session_state.processed   = True
            st.session_state.current_idx = 0
            st.session_state.stats = {
                n: {"correct": 0, "total": 0} for n in models
            }
            st.rerun()
    else:
        n_win = len(st.session_state.all_results)
        st.markdown(f"**{n_win} pencere** hazır")
        st.markdown("---")

        st.markdown("**▶ Oynatma Hızı**")
        speed_map = {"0.5×": 0.5, "1× Gerçek Zamanlı": 1.0,
                     "2×": 2.0, "5×": 5.0}
        sp_label = st.radio("hız", list(speed_map.keys()),
                             index=1, label_visibility="collapsed")
        st.session_state.speed = speed_map[sp_label]

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.playing:
                if st.button("⏸ Durdur", use_container_width=True):
                    st.session_state.playing = False
                    st.rerun()
            else:
                if st.button("▶ Başlat", type="primary",
                             use_container_width=True):
                    st.session_state.playing = True
                    st.rerun()
        with c2:
            if st.button("↩ Sıfırla", use_container_width=True):
                st.session_state.current_idx = 0
                st.session_state.playing     = False
                st.session_state.stats = {
                    n: {"correct": 0, "total": 0} for n in models
                }
                st.rerun()

        st.markdown("---")
        idx = st.slider(
            "Manuel adım", 0,
            max(0, len(st.session_state.all_results) - 1),
            st.session_state.current_idx,
            disabled=st.session_state.playing,
        )
        if idx != st.session_state.current_idx and not st.session_state.playing:
            st.session_state.current_idx = idx
            st.session_state.stats = {
                n: {
                    "correct": sum(1 for r in st.session_state.all_results[:idx+1]
                                   if r["preds"][n]["match"] and r["gt"] != "?"),
                    "total":   sum(1 for r in st.session_state.all_results[:idx+1]
                                   if r["gt"] != "?"),
                }
                for n in models
            }
            st.rerun()


# ── Sonuç paneli ─────────────────────────────────────────────────────────────
if not st.session_state.processed:
    st.info("⬅ **'Ses Dosyasını İşle'** ile başlayın.")
    st.markdown("""
    **Demo sesi:** BÜRKÜT (30s) → DİNOZOR (30s) → HEXA (30s) → ÖRÜMCEK (30s)

    **Pipeline:** 7 seviye DWT (sym4) + TSK-LHP → MI özellik seçimi (35 özellik)

    **Modeller:**
    - 🔵 **kNN** — k=5, Manhattan mesafesi
    - 🟢 **SVM** — kübik poly kernel
    - 🟠 **RF**  — 200 ağaç
    """)
    st.stop()

results  = st.session_state.all_results
cidx     = st.session_state.current_idx
cur      = results[cidx]
gt_cls   = cur["gt"]
t_pos    = cur["time"]
t_end    = cur["time_end"]
total_s  = results[-1]["time_end"]
stats    = st.session_state.stats
model_names = list(ALL_MODELS.keys())

# ── Zaman çubuğu ──────────────────────────────────────────────────────────────
prog_col, time_col = st.columns([5, 1])
with prog_col:
    st.progress(t_pos / total_s if total_s > 0 else 0)
with time_col:
    st.caption(f"t = {t_pos:.1f}s / {total_s:.0f}s")

# ── 3 Model yan yana ─────────────────────────────────────────────────────────
cols = st.columns(3)

for col, m_name in zip(cols, model_names):
    p       = cur["preds"][m_name]
    cls     = p["class"]
    conf    = p["confidence"]
    match   = p["match"]
    probs   = p["probs"]
    m_color = MODEL_LABEL_COLORS.get(m_name, "#cdd6f4")
    c_color = CLASS_COLORS.get(cls, "#7f7f7f")

    with col:
        # Model başlığı
        st.markdown(
            f'<div class="model-header" style="border-bottom:2px solid {m_color};">'
            f'{m_name}</div>',
            unsafe_allow_html=True,
        )

        # Tahmin kutusu
        st.markdown(
            f'<div class="class-box" style="background:{c_color};">{cls}</div>',
            unsafe_allow_html=True,
        )

        # Güven
        st.caption(f"Güven: %{conf*100:.1f}")
        st.progress(conf)

        # Ground truth karşılaştırma
        if gt_cls != "?":
            if match:
                st.markdown(f'<span class="match-ok">✔ {gt_cls}</span>',
                            unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<span class="match-err">✘ Beklenen: {gt_cls}</span>',
                    unsafe_allow_html=True,
                )

        # Sınıf olasılıkları
        st.markdown("---")
        sorted_p = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        for c_name, c_prob in sorted_p:
            c_col = CLASS_COLORS.get(c_name, "#7f7f7f")
            sub1, sub2 = st.columns([1, 2])
            with sub1:
                st.markdown(
                    f'<span style="color:{c_col};font-size:0.75rem;">'
                    f'{c_name}</span>',
                    unsafe_allow_html=True,
                )
            with sub2:
                st.progress(float(c_prob))

st.markdown("---")

# ── Timeline ──────────────────────────────────────────────────────────────────
st.markdown("**Tahmin Geçmişi** — yeşil kenarlık = doğru, kırmızı = yanlış")
st.plotly_chart(make_timeline(results, cidx), use_container_width=True)

st.markdown("---")

# ── Model karşılaştırma tablosu + bar chart ───────────────────────────────────
stat_cols = st.columns([2, 1])

with stat_cols[0]:
    st.markdown("**Model Karşılaştırması (bu ana kadar)**")
    header = "| Model | CV Doğruluk | Split Doğruluk | Demo Doğruluk | Doğru/Toplam |"
    sep    = "|-------|-------------|----------------|---------------|--------------|"
    rows_md = [header, sep]
    for m_name, m_info in st.session_state.models.items():
        s      = stats.get(m_name, {"correct": 0, "total": 0})
        demo_a = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
        cv_a   = m_info.get("cv_accuracy", m_info["accuracy"]) * 100
        sp_a   = m_info["accuracy"] * 100
        rows_md.append(
            f"| **{m_name}** | %{cv_a:.1f} | %{sp_a:.1f} | %{demo_a:.1f} "
            f"| {s['correct']}/{s['total']} |"
        )
    st.markdown("\n".join(rows_md))

with stat_cols[1]:
    st.markdown("**Demo Doğruluk**")
    st.plotly_chart(make_acc_bar(stats), use_container_width=True)

st.caption(
    f"Pencere {cidx+1}/{len(results)}  |  "
    f"t={t_pos:.1f}s–{t_end:.1f}s  |  "
    f"GT: {gt_cls}"
)

# ── Otomatik yenileme ─────────────────────────────────────────────────────────
if st.session_state.playing:
    if cidx < len(results) - 1:
        # İstatistikleri güncelle
        if cur["gt"] != "?":
            for m_name in model_names:
                if cur["preds"][m_name]["match"]:
                    stats[m_name]["correct"] += 1
                stats[m_name]["total"] += 1
        st.session_state.stats        = stats
        st.session_state.current_idx += 1
        time.sleep(HOP_DURATION / st.session_state.speed)
        st.rerun()
    else:
        st.session_state.playing = False
        st.balloons()
