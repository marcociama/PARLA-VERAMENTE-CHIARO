"""
daws/ui/app.py  —  DAWS Dashboard (Streamlit, light theme)
------------------------------------------------------
Dialect-Aware Warning System · PARLA CHIARO · UniNA PICUS Lab
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st

# Project root → two levels up from daws/ui/
BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

st.set_page_config(
    page_title="DAWS — DialectGuard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
_C_GREEN  = "#2ecc71"
_C_YELLOW = "#f39c12"
_C_ORANGE = "#e67e22"
_C_RED    = "#e74c3c"
_C_BLUE   = "#3498db"
_C_GREY   = "#95a5a6"

_RISK_COLOR = {"green": _C_GREEN, "yellow": _C_YELLOW, "red": _C_RED}
_RISK_LABEL = {
    "green":  "RISCHIO BASSO",
    "yellow": "ATTENZIONE — RISCHIO MODERATO",
    "red":    "RISCHIO ELEVATO",
}
_RISK_DESC = {
    "green":  "Trascrizione coerente con il ground truth. Procedere normalmente.",
    "yellow": "Incertezza rilevata. Verificare la trascrizione prima di procedere.",
    "red":    "Rifiutare e richiedere la riformulazione con questionario standard.",
}

_OUTPUTS      = BASE / "outputs_geometrici"
_CFG_PATH     = BASE / "config" / "geometry_calibration.json"
_DATASET_PATH = _OUTPUTS / "dataset_topologico_50.json"

# Medical uncertainty lexicon: lemma → (score [0-1], [interpretazioni])
_MED_LEX: dict[str, tuple[float, list[str]]] = {
    "ambiguo":             (0.90, ["Semantic Shift", "Vague"]),
    "ambigua":             (0.90, ["Semantic Shift", "Vague"]),
    "forse":               (0.88, ["Not Clear", "Equivocal"]),
    "sembrerebbe":         (0.85, ["Equivocal", "Semantic Shift"]),
    "possibile":           (0.82, ["Equivocal", "Not Clear"]),
    "possibilmente":       (0.82, ["Equivocal"]),
    "potrebbe":            (0.80, ["Equivocal", "Context Drift"]),
    "potrebbero":          (0.80, ["Equivocal", "Context Drift"]),
    "sembra":              (0.78, ["Equivocal", "Context Drift"]),
    "eventuale":           (0.75, ["Semantic Shift", "Vague"]),
    "eventualmente":       (0.72, ["Context Drift"]),
    "pare":                (0.75, ["Equivocal"]),
    "probabilmente":       (0.70, ["Equivocal"]),
    "probabile":           (0.70, ["Equivocal"]),
    "circa":               (0.60, ["Vague"]),
    "quasi":               (0.58, ["Vague"]),
    "approssimativamente": (0.65, ["Vague"]),
    "dovrebbe":            (0.65, ["Context Drift"]),
    "dovrebbero":          (0.65, ["Context Drift"]),
    "suggerisce":          (0.60, ["Context Drift", "Semantic Shift"]),
    "indica":              (0.55, ["Semantic Shift"]),
    "valutare":            (0.52, ["Not Clear"]),
    "urgente":             (0.55, ["Semantic Shift"]),
    "talvolta":            (0.55, ["Vague"]),
    "spesso":              (0.50, ["Vague"]),
    "generalmente":        (0.48, ["Context Drift"]),
    "considerare":         (0.50, ["Context Drift"]),
    "verificare":          (0.48, ["Not Clear"]),
    "consultare":          (0.30, ["Standard"]),
    "consiglio":           (0.25, ["Standard"]),
    "raccomando":          (0.22, ["Standard"]),
    "normale":             (0.20, ["Standard"]),
    "regolare":            (0.18, ["Standard"]),
}

_FWORDS = {
    "il","lo","la","i","gli","le","un","una","uno","l",
    "di","a","da","in","con","su","per","tra","fra",
    "e","o","ma","se","che","come","quando","dove",
    "è","sono","ha","hanno","non","si","ci","ne","mi","ti","vi",
    "al","del","nel","dal","col","sul","ai","dei","nei","dai","sui",
    "alla","della","nella","dalla","sulla","alle","delle","nelle","dalle","sulle",
    "questo","questa","questi","queste","quello","quella",
    "suo","sua","suoi","sue","mio","mia",
    "lei","lui","loro","noi","voi",
    "anche","più","molto","poi","già","sempre","ancora","qui","lì",
    "no","sì","ho","sei","era","fu","essere","avere","fare",
}


# ══════════════════════════════════════════════════════════════════════════════
# Tooltip CSS (only CSS injected — no theme override)
# ══════════════════════════════════════════════════════════════════════════════

def _inject_tooltip_css() -> None:
    st.markdown("""
<style>
.daws-tok {
    display: inline-block;
    position: relative;
    cursor: help;
    padding-bottom: 2px;
    vertical-align: bottom;
}
.daws-tok::after {
    content: attr(data-tip);
    white-space: pre;
    display: none;
    position: absolute;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    background: #ffffff;
    color: #212529;
    border: 1px solid #3498db;
    padding: 7px 12px;
    border-radius: 6px;
    font-size: 11.5px;
    font-family: system-ui, sans-serif;
    line-height: 1.65;
    z-index: 99999;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    pointer-events: none;
    min-width: 150px;
}
.daws-tok:hover::after {
    display: block;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Resource loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Caricamento pipeline DAWS (WhisperX · SBERT · Mistral)...")
def _load_pipeline():
    from daws.pipeline.daws import DAWSPipeline
    # use_mongo=False: la UI gestisce la persistenza esplicita con feedback visivo
    return DAWSPipeline(use_mongo=False)


@st.cache_data
def _load_cfg() -> Optional[dict]:
    if not _CFG_PATH.exists():
        return None
    return json.loads(_CFG_PATH.read_text(encoding="utf-8"))


@st.cache_data
def _load_dataset() -> list[dict]:
    if not _DATASET_PATH.exists():
        return []
    return json.loads(_DATASET_PATH.read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════════
# Render helpers
# ══════════════════════════════════════════════════════════════════════════════

def _score_color(s: float) -> str:
    if s < 0.25: return _C_GREEN
    if s < 0.45: return "#82c99a"
    if s < 0.60: return _C_YELLOW
    if s < 0.75: return _C_ORANGE
    return _C_RED


def _risk_badge(risk: str) -> None:
    color = _RISK_COLOR.get(risk, _C_GREY)
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    st.markdown(
        f'<div style="background:{color};color:#fff;padding:16px 28px;'
        f'border-radius:10px;margin:10px 0 18px 0;'
        f'box-shadow:0 4px 18px rgba({r},{g},{b},0.35);">'
        f'<div style="font-size:22px;font-weight:800;letter-spacing:1.5px;">'
        f'{_RISK_LABEL[risk]}</div>'
        f'<div style="font-size:14px;margin-top:6px;opacity:0.92;">'
        f'{_RISK_DESC[risk]}</div></div>',
        unsafe_allow_html=True,
    )


def _heatmap_wrap(inner: str) -> str:
    return (
        '<div style="background:#fff;border:1px solid #dee2e6;'
        'border-radius:8px;padding:16px 20px;line-height:3.0;'
        'font-family:system-ui,sans-serif;font-size:15px;">'
        + inner + "</div>"
    )


def _tok_span(word: str, color: str, tip: str) -> str:
    safe = tip.replace('"', "'")
    return (
        f'<span class="daws-tok" data-tip="{safe}" '
        f'style="border-bottom:4px solid {color};margin:0 1px;">{word}</span>'
    )


def _plain_span(word: str) -> str:
    return f'<span style="margin:0 1px;">{word}</span>'


def _heatmap_legend() -> None:
    items = [
        (_C_GREEN,  "Bassa < 0.25"),
        ("#82c99a", "Lieve 0.25–0.45"),
        (_C_YELLOW, "Moderata 0.45–0.60"),
        (_C_ORANGE, "Elevata 0.60–0.75"),
        (_C_RED,    "Critica > 0.75"),
    ]
    html = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:16px;font-size:12px;">'
        f'<span style="width:22px;height:4px;background:{c};display:inline-block;'
        f'margin-right:5px;border-radius:2px;"></span>{lbl}</span>'
        for c, lbl in items
    )
    st.markdown(f'<div style="margin-top:8px;">{html}</div>', unsafe_allow_html=True)


def _whisperx_heatmap(words: list[dict]) -> str:
    if not words:
        return _heatmap_wrap("<em style='color:#6c757d'>Nessun dato di allineamento.</em>")
    parts = []
    for w in words:
        word = w.get("word", "")
        conf = float(w.get("confidence", 0.5))
        unc  = 1.0 - conf
        col  = _score_color(unc)
        tip  = f"Parola: {word}\nConfidenza: {conf:.3f}\nIncertezza: {unc:.3f}"
        parts.append(_tok_span(word, col, tip))
    return _heatmap_wrap(" ".join(parts))


def _mistral_heatmap(response: str, u_pipeline: float) -> str:
    if not response:
        return _heatmap_wrap("<em style='color:#6c757d'>Nessuna risposta.</em>")
    toks  = re.findall(r"[\w'àèéìòùÀÈÉÌÒÙ]+|[^\w\s]", response)
    bleed = u_pipeline * 0.45
    parts = []
    for tok in toks:
        if re.fullmatch(r"[^\w']+", tok):
            parts.append(_plain_span(tok))
            continue
        key = tok.lower().rstrip(".,;:!?")
        if key in _FWORDS:
            parts.append(_plain_span(tok))
            continue
        if key in _MED_LEX:
            score, interps = _MED_LEX[key]
            interp_str = " · ".join(interps)
        else:
            score = bleed
            interp_str = "Contenuto medico generale"
        if score <= 0.12:
            parts.append(_plain_span(tok))
            continue
        col = _score_color(score)
        tip = f"Token: {tok}\nPunteggio: {score:.2f}\nInterpretazioni: {interp_str}"
        parts.append(_tok_span(tok, col, tip))
    return _heatmap_wrap(" ".join(parts))


def _diff_heatmap(gt_text: str, w_text: str) -> None:
    """
    Side-by-side Grammarly-style diff heatmap.

    Coloring for replace blocks uses character-level similarity between
    the best-matching word pair, so completely different words → red,
    near-misses → orange, partial matches → yellow.
    Equal blocks → green. Delete/insert → red.
    """
    def _clean(s: str) -> str:
        return re.sub(r"[^a-zàèéìòù0-9]", "", s.lower())

    def _sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    gt_toks = gt_text.split()
    w_toks  = w_text.split()
    gt_keys = [_clean(t) for t in gt_toks]
    w_keys  = [_clean(t) for t in w_toks]

    matcher   = difflib.SequenceMatcher(None, gt_keys, w_keys, autojunk=False)
    gt_colors = [_C_GREEN] * len(gt_toks)
    gt_tips   = [f"{t}\nCorretta corrispondenza" for t in gt_toks]
    w_colors  = [_C_GREEN] * len(w_toks)
    w_tips    = [f"{t}\nCorretta corrispondenza" for t in w_toks]

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pass   # already green

        elif tag == "delete":
            for i in range(i1, i2):
                gt_colors[i] = _C_RED
                gt_tips[i]   = f"{gt_toks[i]}\nEliminata dall'ASR"

        elif tag == "insert":
            for j in range(j1, j2):
                w_colors[j] = _C_RED
                w_tips[j]   = f"{w_toks[j]}\nInserita dall'ASR (assente nel GT)"

        elif tag == "replace":
            gk = gt_keys[i1:i2]
            wk = w_keys[j1:j2]
            # GT side: color each GT word by its best match in W block
            for ii, (tok, key) in enumerate(zip(gt_toks[i1:i2], gk)):
                best = max((_sim(key, wkey) for wkey in wk), default=0.0)
                if best >= 0.70:
                    col, lbl = _C_YELLOW, f"Sostituzione parziale  sim={best:.2f}"
                elif best >= 0.35:
                    col, lbl = _C_ORANGE, f"Sostituzione distante  sim={best:.2f}"
                else:
                    col, lbl = _C_RED,    f"Eliminata/Diversa  sim={best:.2f}"
                gt_colors[i1 + ii] = col
                gt_tips  [i1 + ii] = f"{tok}\n{lbl}"
            # W side: color each W word by its best match in GT block
            for jj, (tok, key) in enumerate(zip(w_toks[j1:j2], wk)):
                best = max((_sim(key, gkey) for gkey in gk), default=0.0)
                if best >= 0.70:
                    col, lbl = _C_YELLOW, f"Sostituzione parziale  sim={best:.2f}"
                elif best >= 0.35:
                    col, lbl = _C_ORANGE, f"Sostituzione distante  sim={best:.2f}"
                else:
                    col, lbl = _C_RED,    f"Inserita/Diversa  sim={best:.2f}"
                w_colors[j1 + jj] = col
                w_tips  [j1 + jj] = f"{tok}\n{lbl}"

    def _render(toks, colors, tips):
        parts = [_tok_span(tok, col, tip)
                 for tok, col, tip in zip(toks, colors, tips)]
        return _heatmap_wrap(" ".join(parts))

    cgt, cw = st.columns(2)
    cgt.caption("GROUND TRUTH (GT1)")
    cgt.markdown(_render(gt_toks, gt_colors, gt_tips), unsafe_allow_html=True)
    cw.caption("TRASCRIZIONE ASR (W1)")
    cw.markdown(_render(w_toks, w_colors, w_tips), unsafe_allow_html=True)

    legend_items = [
        (_C_GREEN,  "Corrisponde"),
        (_C_YELLOW, "Sostituzione parziale (sim ≥ 0.70)"),
        (_C_ORANGE, "Sostituzione distante (sim 0.35–0.70)"),
        (_C_RED,    "Completamente diversa (sim < 0.35)"),
    ]
    html = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;font-size:12px;">'
        f'<span style="width:18px;height:4px;background:{c};display:inline-block;'
        f'margin-right:4px;border-radius:2px;"></span>{lbl}</span>'
        for c, lbl in legend_items
    )
    st.markdown(f'<div style="margin-top:8px;">{html}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Plotly charts — no shared layout dict, params set directly to avoid conflicts
# ══════════════════════════════════════════════════════════════════════════════

def _drift_fig(s_w: list[float], cfg: dict,
               transcript: str = "", llm_response: str = "",
               h_risk: float = 0.0, risk_level: str = "green"):
    import plotly.graph_objects as go

    mu_gt    = [cfg["mu_R_GT1"], cfg["mu_R_GT2"], cfg["mu_R_GT3"]]
    mu_mean  = float(np.mean(mu_gt))
    gt_span  = max(mu_gt) - min(mu_gt)
    if gt_span < 1e-6:
        gt_span = max(abs(mu_mean) * 0.1, 0.01)

    # Zone boundaries — geometric context only (not predictive of H_risk)
    green_half  = 1.5 * gt_span
    yellow_half = 3.5 * gt_span

    # Axis range: fit data + 20% margin; red extends beyond visible range
    all_x   = list(s_w) + list(mu_gt)
    x_lo    = min(all_x) - 3 * gt_span
    x_hi    = max(all_x) + 3 * gt_span
    far_pad = (x_hi - x_lo) * 50   # extends well beyond axis range → "infinite"

    # Point and spread-bar color = H_risk classification (matches risk badge)
    risk_color = {
        "green":  _C_GREEN,
        "yellow": _C_YELLOW,
        "red":    _C_RED,
    }.get(risk_level, _C_RED)

    tr_s = (transcript[:65]+"…")   if len(transcript)>65   else transcript
    r_s  = (llm_response[:65]+"…") if len(llm_response)>65 else llm_response

    fig = go.Figure()

    # Background geometric bands (position-based context — may disagree with H_risk)
    fig.add_vrect(x0=mu_mean - far_pad,     x1=mu_mean - yellow_half,
                  fillcolor=_C_RED,    opacity=0.05, layer="below", line_width=0)
    fig.add_vrect(x0=mu_mean + yellow_half, x1=mu_mean + far_pad,
                  fillcolor=_C_RED,    opacity=0.05, layer="below", line_width=0)
    fig.add_vrect(x0=mu_mean - yellow_half, x1=mu_mean - green_half,
                  fillcolor=_C_YELLOW, opacity=0.07, layer="below", line_width=0)
    fig.add_vrect(x0=mu_mean + green_half,  x1=mu_mean + yellow_half,
                  fillcolor=_C_YELLOW, opacity=0.07, layer="below", line_width=0)
    fig.add_vrect(x0=mu_mean - green_half,  x1=mu_mean + green_half,
                  fillcolor=_C_GREEN,  opacity=0.06, layer="below", line_width=0)

    # GT anchors
    fig.add_trace(go.Scatter(
        x=mu_gt, y=[0]*3, mode="markers+text",
        marker=dict(size=18, color=_C_BLUE, symbol="circle",
                    line=dict(width=2, color="#2980b9")),
        name="Ancore GT (calibrazione)",
        text=[f"μ_GT{i+1}" for i in range(3)],
        textposition="top center",
        textfont=dict(size=11, color=_C_BLUE),
        hovertext=[f"μ_GT{i+1} — ancora calibrazione\nScalare: {v:.5f}"
                   for i, v in enumerate(mu_gt)],
        hoverinfo="text",
    ))

    # TTA spread bar — width = dispersion of W1/W2/W3, colour = H_risk
    if len(s_w) >= 2:
        sw_min, sw_max = min(s_w), max(s_w)
        sw_mid = float(np.mean(s_w))
        spread = sw_max - sw_min
        fig.add_trace(go.Scatter(
            x=[sw_min, sw_max], y=[0, 0],
            mode="lines",
            line=dict(color=risk_color, width=6),
            name=f"Dispersione TTA  Δ={spread:.5f}  H_risk={h_risk:.3f}",
            hovertext=[f"Dispersione TTA  Δ={spread:.5f}  H_risk={h_risk:.3f}"] * 2,
            hoverinfo="text",
            showlegend=True,
        ))

    # W projection points — coloured by H_risk (matches badge)
    w_tips = ([
        f"s_W1 — risposta primaria\nScalare: {s_w[0]:.5f}\n{tr_s}\n→ {r_s}",
        f"s_W2 — TTA seed 1\nScalare: {s_w[1]:.5f}",
        f"s_W3 — TTA seed 2\nScalare: {s_w[2]:.5f}",
    ] if len(s_w) >= 3 else [f"s_W{i+1}\nScalare: {v:.5f}" for i, v in enumerate(s_w)])

    fig.add_trace(go.Scatter(
        x=s_w, y=[0]*len(s_w), mode="markers+text",
        marker=dict(size=15, color=risk_color, symbol="diamond",
                    line=dict(width=1.5, color="#555")),
        name="Proiezioni live (s_W)",
        text=[f"s_W{i+1}" for i in range(len(s_w))],
        textposition="bottom center",
        textfont=dict(size=11, color=risk_color),
        hovertext=w_tips, hoverinfo="text",
    ))

    fig.update_layout(
        title=dict(
            text="Proiezione 1D — asse di deriva semantica  GT ↔ ASR  "
                 "(sfondo: distanza geometrica | punti/barra: H_risk)",
            font=dict(size=12)),
        xaxis=dict(title="Coordinata scalare su w_resp_drift",
                   showgrid=True, gridcolor="#f0f0f0",
                   range=[x_lo, x_hi]),
        yaxis=dict(visible=False, range=[-0.8, 0.8]),
        height=240,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=55, b=40),
    )
    return fig


def _scatter_fig(xs, ys, labels, dialects, x_title, y_title, title, r_val):
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    df = pd.DataFrame({"x": xs, "y": ys, "stem": labels, "dialetto": dialects})
    fig = px.scatter(
        df, x="x", y="y", color="dialetto",
        hover_data={"stem": True, "x": ":.4f", "y": ":.4f", "dialetto": True},
        labels={"x": x_title, "y": y_title},
        title=f"{title}  (r = {r_val:+.2f}, n={len(xs)})",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    if len(xs) > 2:
        m, b = np.polyfit(xs, ys, 1)
        xl = [min(xs), max(xs)]
        fig.add_trace(go.Scatter(
            x=xl, y=[m*v+b for v in xl], mode="lines",
            line=dict(dash="dash", color=_C_GREY, width=1.5),
            name=f"OLS  y={m:.3f}x+{b:.3f}", hoverinfo="skip",
        ))
    fig.update_layout(
        height=420, hovermode="closest",
        margin=dict(l=10, r=10, t=50, b=40),
        xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Historical corpus fallback
# ══════════════════════════════════════════════════════════════════════════════

def _load_historical_fallback() -> list[dict]:
    dataset = _load_dataset()
    if not dataset:
        return []
    out = []
    for d in dataset:
        h_k1    = d.get("inv_entropy", {}).get("H_k1", 0.0)
        h_sp    = float(d.get("h_spectral", 0.0))
        h_risk  = _h_risk_from_spectral(h_sp)
        pca     = d.get("pca_local", {})
        txts    = d.get("texts", [])
        wer     = round(d.get("wer", 0.0), 4)
        dpc1    = round(pca.get("delta_pc1", 0.0), 4)
        out.append({
            "_source":        "Historical Corpus",
            "_display_id":    d.get("stem", ""),
            "dialect":        d.get("dialect", ""),
            "gender":         d.get("gender", ""),
            "age_range":      d.get("age_range", ""),
            "wer":            wer,
            "H_k6":           round(h_k1, 4),
            "h_spectral":     round(h_sp, 5),
            "h_risk":         round(h_risk, 4),
            "severity_score": round(h_risk * (wer + dpc1), 4),
            "delta_pc1":      dpc1,
            "gt1":            txts[0] if txts else "",
            "w1":             txts[3] if len(txts) > 3 else "",
            "risk_level":     _risk_from_h_risk(h_risk),
        })
    return out


_H_SP_MIN = 0.4320   # bounds empirici N=50 (ablation 2026-05-21)
_H_SP_MAX = 0.8452
_H_SP_RNG = _H_SP_MAX - _H_SP_MIN  # 0.4132


def _h_risk_from_spectral(h_sp: float) -> float:
    """Normalizza H_spectral nei bounds empirici → H_risk ∈ [0,1]."""
    return float(np.clip((h_sp - _H_SP_MIN) / max(_H_SP_RNG, 1e-12), 0.0, 1.0))


def _risk_from_h_risk(h_risk: float) -> str:
    if h_risk < 0.39:  return "green"
    if h_risk < 0.52:  return "yellow"
    return "red"


def _seed_corpus_to_mongo(repo, dataset: list[dict]) -> int:
    """Bulk-upsert N=50 corpus records into MongoDB recordings.
    risk_level = _risk_from_h_risk(H_risk(h_spectral)) — soglie ROC-calibrate 0.39/0.52.
    Returns count of upserted+modified documents."""
    if not dataset or not repo._connect():
        return 0

    from datetime import datetime, timezone
    from pymongo import UpdateOne

    # Build u_asr lookup from ASR cache (keyed by stem = filename without extension)
    _asr_cache_dir = BASE / "daws" / "results" / "asr_cache"
    _u_asr_lookup: dict[str, float] = {}
    if _asr_cache_dir.exists():
        import json as _json
        for _f in _asr_cache_dir.glob("*.json"):
            try:
                _d = _json.loads(_f.read_text())
                _u_asr_lookup[_f.stem] = float(_d.get("u_asr", 0.0))
            except Exception:
                pass

    now = datetime.now(timezone.utc)

    ops = []
    for d in dataset:
        h_k1    = d.get("inv_entropy", {}).get("H_k1", 0.0)
        h_sp    = float(d.get("h_spectral", 0.0))
        h_risk  = _h_risk_from_spectral(h_sp)
        pca     = d.get("pca_local", {})
        txts    = d.get("texts", [])
        wer     = float(d.get("wer", 0.0))
        dpc1    = float(pca.get("delta_pc1", 0.0))
        stem    = d.get("stem", "")
        u_asr   = _u_asr_lookup.get(stem, 0.0)
        ops.append(UpdateOne(
            {"filename": stem},
            {"$set": {
                "filename":         stem,
                "dialect":          d.get("dialect", ""),
                "gender":           d.get("gender", ""),
                "age_range":        d.get("age_range", ""),
                "wer":              wer,
                "u_asr":            u_asr,
                "H_k6":             float(h_k1),
                "h_spectral":       h_sp,
                "h_risk":           h_risk,
                "severity_score":   round(h_risk * (wer + dpc1), 4),
                "delta_pc1":        dpc1,
                "gt1":              txts[0] if txts else "",
                "w1":               txts[3] if len(txts) > 3 else "",
                "risk_level":       _risk_from_h_risk(h_risk),
                "created_at":       now,
            }},
            upsert=True,
        ))

    try:
        res = repo._db["recordings"].bulk_write(ops, ordered=False)
        return res.upserted_count + res.modified_count
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# Section: Inferenza
# ══════════════════════════════════════════════════════════════════════════════

def _section_inferenza() -> None:
    st.header("Inferenza")
    st.markdown(
        "Registra la dichiarazione del paziente dal microfono oppure carica un file WAV. "
        "WhisperX trascrive con 3-way TTA, Mistral 7B genera le risposte e "
        "H_spectral (1D Markov Spettrale, Laplace) stima il rischio clinico."
    )

    # Dual audio input
    col_mic, col_sep, col_file = st.columns([5, 1, 5])
    with col_mic:
        st.markdown("**Registra dal microfono**")
        audio_mic = st.audio_input("Premi per registrare", label_visibility="collapsed")
    with col_sep:
        st.markdown("<div style='text-align:center;padding-top:32px;color:#aaa;'>oppure</div>",
                    unsafe_allow_html=True)
    with col_file:
        st.markdown("**Carica un file WAV**")
        audio_file = st.file_uploader("WAV upload", type=["wav"],
                                      label_visibility="collapsed")

    audio_source = audio_mic if audio_mic is not None else audio_file
    if audio_source is None:
        st.info("Registra dal microfono oppure carica un file WAV per avviare l'analisi.")
        return

    if not st.button("Analizza", type="primary"):
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_source.read())
        tmp_path = tmp.name

    with st.spinner("Analisi in corso — trascrizione · TTA · risposte Mistral · H_spectral..."):
        try:
            result = _load_pipeline().process_audio(tmp_path)
        except Exception as exc:
            st.error(f"Errore pipeline: {exc}")
            return

    ui_risk = result.risk_level   # calibrated P33/P66 thresholds from pipeline

    st.session_state["last_risk"]       = ui_risk
    st.session_state["last_u_pipeline"] = result.u_pipeline

    # Persist to MongoDB (UI-layer save with visible feedback)
    _mongo_saved = False
    _mongo_err   = ""
    try:
        from daws.database import DAWSRepository
        _repo_ui = DAWSRepository()
        _inf_id  = _repo_ui.log_inference({
            "transcript":          result.transcript,
            "u_asr":               float(result.u_asr),
            "u_llm":               float(result.u_llm),
            "u_pipeline":          float(result.u_pipeline),
            "h_spectral":          float(result.h_spectral),
            "h_risk":              float(result.h_risk),
            "risk_level":          ui_risk,
            "llm_response":        result.llm_response,
            "clarifying_question": result.clarification_question,
        })
        _mongo_saved = bool(_inf_id)
    except Exception as _exc:
        _mongo_err = str(_exc)

    # Risk badge
    st.markdown("---")
    if _mongo_saved:
        st.success("Inferenza salvata su MongoDB — visibile in Audit Trail (fonte: Live Production).")
    else:
        st.caption(
            "MongoDB non raggiungibile — l'inferenza non è stata persistita."
            + (f" ({_mongo_err})" if _mongo_err else "")
        )
    _risk_badge(ui_risk)

    if ui_risk == "red" and result.clarification_question:
        st.error(f"**Domanda di chiarimento:** {result.clarification_question}")

    # Metrics — solo valori direttamente calcolabili a runtime (no WER, no ΔPC1 → no Severity Score)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("U_ASR",      f"{result.u_asr:.3f}",
              help="1 − confidenza media WhisperX (proxy acustico del WER)")
    m2.metric("H_spectral", f"{result.h_spectral:.4f} nat",
              help="Entropia spettrale 1D Markov Laplace (nats). "
                   "Range calibrazione N=50: [0.4320, 0.8452]. "
                   "Valori inferiori al minimo → drift sotto soglia corpus.")
    m3.metric("H_risk",     f"{result.h_risk:.3f}",
              help="(H_spectral − 0.4320) / 0.4132  ∈ [0,1] "
                   "| VERDE < 0.39 | GIALLO < 0.52 | ROSSO ≥ 0.52 "
                   "(soglie ROC-calibrate su N=50, AUC_red=0.85)")
    m4.metric("Tempo",      f"{result.processing_time_s:.1f} s")

    if result.h_spectral < 0.4320:
        st.caption(
            f"H_spectral = {result.h_spectral:.4f} nats — sotto il minimo del corpus di calibrazione "
            "(0.4320 nats, N=50). Deriva semantica inferiore al caso meno dialettale osservato. "
            "Il Severity Score (H_risk × (WER + |ΔPC1|)) è calcolato nell'Audit Trail "
            "dove WER e ΔPC1 sono disponibili."
        )

    # 1D Drift
    cfg = _load_cfg()
    if cfg and result.s_w:
        st.subheader("Proiezione 1D — asse di deriva semantica")
        st.plotly_chart(
            _drift_fig(result.s_w, cfg, result.transcript, result.llm_response,
                       h_risk=result.h_risk, risk_level=ui_risk),
            use_container_width=True,
        )
        st.caption(
            "Cerchi blu = ancore μ_GT (calibrazione N=50). "
            "Diamanti + barra = proiezioni live s_W1/W2/W3, colorati per **H_risk** (coerente col badge). "
            "Sfondo = distanza geometrica da μ_GT (contesto, non predice H_risk)."
        )

    # Token heatmap — WhisperX acoustic alignment only
    st.subheader("Allineamento acustico WhisperX")
    st.caption("Barra colorata sotto ogni parola = incertezza ASR (1 − confidenza). "
               "Passa il cursore per il dettaglio.")
    if result.words:
        st.markdown(_whisperx_heatmap(result.words), unsafe_allow_html=True)
        _heatmap_legend()
    else:
        st.info("Nessun dato di allineamento parola disponibile.")

    # Transcript
    st.subheader("Trascrizione (W1)")
    st.write(result.transcript)

    if result.llm_response:
        st.subheader("Risposta Mistral 7B (W1)")
        st.write(result.llm_response)


# ══════════════════════════════════════════════════════════════════════════════
# Section: Audit Trail
# ══════════════════════════════════════════════════════════════════════════════

def _section_audit() -> None:
    import pandas as pd

    st.header("Audit Trail")
    st.markdown(
        "Vista unificata: corpus storico N=50 (calibrazione) "
        "e inferenze live dalla pipeline in produzione."
    )

    recs_raw: list[dict] = []
    infs_raw: list[dict] = []
    _mongo_repo = None

    try:
        from daws.database import DAWSRepository
        _mongo_repo = DAWSRepository()
        if _mongo_repo._connect():
            recs_raw = _mongo_repo.get_recordings(limit=200)
            infs_raw = _mongo_repo.get_inferences(limit=100)
    except Exception:
        pass

    # Auto-seed: MongoDB raggiungibile ma recordings vuota, oppure dati stale (u_asr tutti 0)
    _needs_seed = _mongo_repo is not None and (
        not recs_raw
        or (recs_raw and all(float(r.get("u_asr", 0.0)) == 0.0 for r in recs_raw))
    )
    if _needs_seed:
        _ds = _load_dataset()
        if _ds and _mongo_repo._connect():
            with st.spinner("Prima apertura: caricamento corpus PARLA CHIARO in MongoDB…"):
                _n_seeded = _seed_corpus_to_mongo(_mongo_repo, _ds)
            if _n_seeded > 0:
                try:
                    recs_raw = _mongo_repo.get_recordings(limit=200)
                    st.success(f"Corpus PARLA CHIARO ({_n_seeded} campioni) caricato in MongoDB.")
                except Exception:
                    pass

    # Annota fonte e risk_level usando H_risk(h_spectral) con soglie ROC-calibrate 0.39/0.52
    if recs_raw:
        for r in recs_raw:
            r["_source"]     = "Historical Corpus"
            r["_display_id"] = r.get("filename", "")
            if "risk_level" not in r:
                h_sp = float(r.get("h_spectral", r.get("H_k6", 0.0)))
                r["risk_level"] = _risk_from_h_risk(_h_risk_from_spectral(h_sp))
            # aggiungi severity_score se manca
            if "severity_score" not in r:
                h_risk = _h_risk_from_spectral(float(r.get("h_spectral", 0.0)))
                wer    = float(r.get("wer", 0.0))
                dpc1   = float(r.get("delta_pc1", 0.0))
                r["severity_score"] = round(h_risk * (wer + dpc1), 4)

    # Fallback JSON solo se MongoDB non disponibile
    if not recs_raw:
        recs_raw = _load_historical_fallback()
        if recs_raw:
            st.caption("MongoDB non disponibile — corpus caricato da dataset_topologico_50.json.")

    # Manual re-seed button (forza ricalcolo risk_level da H_spectral + u_asr reali)
    if _mongo_repo is not None and getattr(_mongo_repo, "_available", None) is True:
        if st.button("🔄 Re-seed corpus", help="Ricalcola risk_level, h_risk, u_asr e severity_score per tutti i 50 record storici usando la pipeline 1D Markov Spettrale corrente."):
            _ds = _load_dataset()
            if _ds:
                with st.spinner("Re-seed in corso…"):
                    _n = _seed_corpus_to_mongo(_mongo_repo, _ds)
                recs_raw = _mongo_repo.get_recordings(limit=200)
                st.success(f"Re-seed completato: {_n} record aggiornati.")
                st.rerun()

    for i in infs_raw:
        i["_source"]     = "Live Production"
        i["_display_id"] = (i.get("transcript") or "")[:60]

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        source_sel = st.multiselect("Fonte dati",
            ["Historical Corpus", "Live Production"],
            default=["Historical Corpus", "Live Production"])
    with fc2:
        risk_sel = st.multiselect("Livello di rischio",
            ["green", "yellow", "red"], default=[],
            help="Lascia vuoto per tutti i livelli.")

    pool: list[dict] = []
    if "Historical Corpus" in source_sel:
        pool.extend(recs_raw)
    if "Live Production" in source_sel:
        pool.extend(infs_raw)
    if risk_sel:
        pool = [d for d in pool if d.get("risk_level") in risk_sel]

    if not pool:
        st.info("Nessun record trovato con i filtri selezionati.")
        return

    # Historical Corpus: count from loaded recs_raw (JSON or MongoDB recordings)
    from collections import Counter
    hist_c = Counter(
        r.get("risk_level", "") for r in recs_raw
        if r.get("risk_level") in ("green", "yellow", "red")
    )

    # Live Production: $group aggregation — riusa _mongo_repo, nessun timeout aggiuntivo
    live_c: dict[str, int] = {"green": 0, "yellow": 0, "red": 0}
    if "Live Production" in source_sel:
        _live_mongo = False
        if _mongo_repo is not None and getattr(_mongo_repo, "_available", None) is True:
            try:
                for doc in _mongo_repo._db["inferences"].aggregate([
                    {"$group": {"_id": "$risk_level", "n": {"$sum": 1}}}
                ]):
                    lvl = doc.get("_id") or ""
                    if lvl in live_c:
                        live_c[lvl] += int(doc["n"])
                _live_mongo = True
            except Exception:
                pass
        if not _live_mongo:
            for _i in infs_raw:
                lvl = _i.get("risk_level", "")
                if lvl in live_c:
                    live_c[lvl] += 1

    n_g = (hist_c["green"]  if "Historical Corpus" in source_sel else 0) + live_c["green"]
    n_y = (hist_c["yellow"] if "Historical Corpus" in source_sel else 0) + live_c["yellow"]
    n_r = (hist_c["red"]    if "Historical Corpus" in source_sel else 0) + live_c["red"]
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Totale", n_g + n_y + n_r)
    s2.metric("Verde",  n_g)
    s3.metric("Giallo", n_y)
    s4.metric("Rosso",  n_r)

    # DataFrame + Styler
    COLS = ["_source", "risk_level", "_display_id", "dialect",
            "gender", "age_range", "wer", "delta_pc1",
            "H_k6", "h_spectral", "severity_score",
            "u_asr", "created_at"]
    df = pd.DataFrame(pool)
    if "created_at" in df.columns:
        df["created_at"] = (pd.to_datetime(df["created_at"], errors="coerce")
                              .dt.strftime("%Y-%m-%d %H:%M"))

    display_cols = [c for c in COLS if c in df.columns]
    rename = {
        "_source":        "Fonte",
        "risk_level":     "Rischio",
        "_display_id":    "ID / Trascrizione",
        "dialect":        "Dialetto",
        "gender":         "Sesso",
        "age_range":      "Età",
        "wer":            "WER",
        "delta_pc1":      "ΔPC1",
        "H_k6":           "Baseline Offline (NeurIPS 2025)",
        "h_spectral":     "Framework Online (1D Markov Spettrale)",
        "severity_score": "Severity Score",
        "u_asr":          "U_ASR",
        "created_at":     "Data",
    }
    df_view = df[display_cols].rename(columns=rename)

    _RISK_CSS = {
        "green":  "background-color:#d5f5e3;color:#1e8449;font-weight:700",
        "yellow": "background-color:#fef9e7;color:#9a7d0a;font-weight:700",
        "red":    "background-color:#fdedec;color:#922b21;font-weight:700",
    }
    styled = (df_view.style
              .map(lambda v: _RISK_CSS.get(v, ""), subset=["Rischio"])
              .format(na_rep="—", precision=4))
    st.dataframe(styled, use_container_width=True, height=450)

    csv = df_view.to_csv(index=False).encode("utf-8")
    st.download_button("Scarica CSV", csv, "daws_audit.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# Section: Analytics
# ══════════════════════════════════════════════════════════════════════════════

def _section_analytics() -> None:
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    st.header("Analytics — Ablation Study N=50 (21 Maggio 2026)")
    st.markdown(
        "**1D Spectral Markov (no GT):** Pearson(WER) = **+0.501** | "
        "Pearson(E_sem_top) = **+0.569** | Pearson(|ΔPC1|) = **+0.43**"
    )

    dataset = _load_dataset()
    tab_ov, tab_pca, tab_cases = st.tabs(["Overview", "Spectral PCA", "Critical Cases"])

    # ── Overview ──────────────────────────────────────────────────────
    with tab_ov:
        if not dataset:
            st.warning("dataset_topologico_50.json non trovato.")
        else:
            wer  = [d.get("wer", 0.0)                             for d in dataset]
            dpc  = [d.get("pca_local",  {}).get("delta_pc1", 0.0) for d in dataset]
            stem = [d.get("stem", "")                              for d in dataset]
            dial = [d.get("dialect", "altro")                      for d in dataset]
            h_sp = [d.get("h_spectral", 0.0)                       for d in dataset]

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    _scatter_fig(wer, h_sp, stem, dial,
                                 "WER", "H_spectral (nats)",
                                 "1D Spectral Markov vs WER  [r=+0.501]",
                                 float(np.corrcoef(wer, h_sp)[0, 1])),
                    use_container_width=True)
            with c2:
                st.plotly_chart(
                    _scatter_fig(dpc, h_sp, stem, dial,
                                 "|ΔPC1|", "H_spectral (nats)",
                                 "1D Spectral Markov vs ΔPC1",
                                 float(np.corrcoef(dpc, h_sp)[0, 1])),
                    use_container_width=True)
            st.caption("|ΔPC1| = distanza centroidi GT↔ASR sul primo asse PCA locale (media 0.55 ± 0.34).")

        cfg = _load_cfg()
        if cfg:
            st.markdown("---")
            st.subheader("Parametri 1D Markov Spettrale — calibrazione N=50")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("σ_in",             f"{cfg.get('sigma_in',  0):.5f}")
            g2.metric("σ_out",            f"{cfg.get('sigma_out', 0):.5f}")
            g3.metric("H_min (nats)",      "0.4320",
                      help="Empirical minimum H_spectral su N=50 (bounds normalizzazione H_risk)")
            g4.metric("H_max (nats)",      "0.8452",
                      help="Empirical maximum H_spectral su N=50")
            g5, g6, g7, g8 = st.columns(4)
            g5.metric("Pearson(H_sp, WER)",  f"{cfg.get('pearson_H_wer', 0.501):.4f}",
                      help="1D Markov Spettrale ONLINE — ablation 2026-05-21")
            g6.metric("H_risk soglia verde",  "0.39",
                      help="H_risk < 0.39 → Rischio Basso (ROC-calibrato, AUC=0.76, sens≥0.90)")
            g7.metric("H_risk soglia rosso",  "0.52",
                      help="H_risk ≥ 0.52 → Rischio Elevato (ROC-calibrato, AUC=0.85, Youden's J)")
            g8.metric("N campioni",            cfg.get("n_samples", 50))

    # ── Spectral PCA ──────────────────────────────────────────────────
    with tab_pca:
        if not dataset:
            st.warning("Dataset non disponibile.")
        else:
            rows = []
            for d in dataset:
                for pt in d.get("pca_local", {}).get("points_pc1_pc2", []):
                    lbl = pt.get("label", "")
                    rows.append({
                        "stem":    d.get("stem", ""),
                        "dialect": d.get("dialect", ""),
                        "wer":     d.get("wer", 0.0),
                        "H_k1":   d.get("inv_entropy", {}).get("H_k1", 0.0),
                        "pc1":     pt.get("pc1", 0.0),
                        "pc2":     pt.get("pc2", 0.0),
                        "text":   (pt.get("text","")[:60]+"…")
                                  if len(pt.get("text",""))>60 else pt.get("text",""),
                        "group":  "GT" if lbl.startswith("GT") else "ASR",
                    })
            if rows:
                df_pca = pd.DataFrame(rows)
                fig_pca = px.scatter(
                    df_pca, x="pc1", y="pc2", color="group",
                    symbol="group",
                    hover_data={"stem": True, "dialect": True, "wer": ":.3f",
                                "H_k1": ":.4f", "text": True,
                                "pc1": ":.4f", "pc2": ":.4f", "group": False},
                    labels={"pc1": "PC1 (locale)", "pc2": "PC2 (locale)",
                            "group": "Origine"},
                    title="PCA locale — tutti i 50 campioni (6 punti/campione)",
                    color_discrete_map={"GT": _C_BLUE, "ASR": _C_RED},
                    symbol_map={"GT": "circle", "ASR": "diamond"},
                    opacity=0.75,
                )
                fig_pca.update_layout(
                    height=500,
                    hovermode="closest",
                    margin=dict(l=10, r=10, t=50, b=40),
                    xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                    yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
                )
                st.plotly_chart(fig_pca, use_container_width=True)
                st.caption("PC1 locale spiega in media il 79% ± 14% della varianza. "
                           "La separazione GT↔ASR sull'asse PC1 riflette la distanza semantica.")

    # ── Critical Cases ────────────────────────────────────────────────
    with tab_cases:
        if not dataset:
            st.warning("Dataset non disponibile.")
            return

        cfg_cal = _load_cfg() or {}
        h_sp_min = float(cfg_cal.get("h_spectral_min", 0.0))
        h_sp_max = float(cfg_cal.get("h_spectral_max", 1.0))
        h_sp_range = max(h_sp_max - h_sp_min, 1e-12)

        def _score(d):
            h_sp   = d.get("h_spectral", 0.0)
            h_risk = float(np.clip((h_sp - h_sp_min) / h_sp_range, 0.0, 1.0))
            return h_risk * (d.get("wer", 0.0)
                             + abs(d.get("pca_local", {}).get("delta_pc1", 0.0)))

        ranked  = sorted(dataset, key=_score, reverse=True)
        max_sc  = _score(ranked[0]) if ranked else 1.0

        # edge case: max is zero - e.g. empty dataset
        if max_sc == 0.0:
            max_sc = 1.0

        # Top-10 casi critici — righe compatte HTML + analisi lazy in expander
        st.subheader("Top 10 casi critici")
        st.caption("Clicca su una riga per espandere l'analisi acustica e il diff testuale GT↔ASR.")

        _DIALECT_COLOR = {
            "napoletano": "#8e44ad", "parmigiano": "#16a085",
            "salentino": "#d35400", "lucano": "#2980b9", "altro": "#7f8c8d",
        }

        for rank, d in enumerate(ranked[:10], 1):
            txts   = d.get("texts", [])
            gt1    = txts[0] if txts else ""
            w1     = txts[3] if len(txts) > 3 else ""
            stem   = d.get("stem", "?")
            wer    = d.get("wer", 0.0)
            h      = d.get("h_spectral", 0.0)
            h_risk = float(np.clip((h - h_sp_min) / h_sp_range, 0.0, 1.0))
            pca    = d.get("pca_local", {})
            dpc    = pca.get("delta_pc1", 0.0)
            sc     = _score(d)
            dial   = d.get("dialect", "altro")
            gender = d.get("gender", "?")
            age    = d.get("age_range", "?")
            sev    = _score_color(sc / max_sc)
            dcol   = _DIALECT_COLOR.get(dial, "#7f8c8d")

            # Compact flat row
            st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;
            border-left:4px solid {sev};background:#fafafa;
            border-radius:0 6px 6px 0;margin:3px 0;">
  <span style="font-weight:700;font-size:13px;color:#6c757d;min-width:22px;">#{rank}</span>
  <span style="font-weight:600;font-size:13px;color:#212529;flex:1;
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{stem}</span>
  <span style="background:{dcol};color:#fff;padding:2px 8px;
               border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap;">{dial.upper()}</span>
  <span style="background:#6c757d;color:#fff;padding:2px 7px;
               border-radius:10px;font-size:11px;white-space:nowrap;">{gender} · {age}</span>
  <span style="background:{sev};color:#fff;padding:3px 10px;
               border-radius:10px;font-size:12px;font-weight:700;white-space:nowrap;">{sc:.3f}</span>
</div>""", unsafe_allow_html=True)

            with st.expander(f"Analisi dettagliata — #{rank} {stem}", expanded=False):
                bc1, bc2, bc3 = st.columns(3)
                bc1.metric("WER",    f"{wer:.3f}")
                bc2.metric("H_risk", f"{h_risk:.3f}", help=f"H_spectral = {h:.5f}")
                bc3.metric("ΔPC1",   f"{dpc:.4f}",
                           help="Distanza GT↔ASR sul primo asse PCA locale")
                if gt1 and w1:
                    _diff_heatmap(gt1, w1)
                else:
                    st.info("Testo non disponibile.")
                st.caption(
                    f"PC1 var.: {pca.get('pc1_variance_explained', 0):.1%} · "
                    f"Score = H_risk × (WER + |ΔPC1|) = {sc:.4f}"
                )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> str:
    st.sidebar.title("DAWS — DialectGuard")
    st.sidebar.markdown(
        "**Dialect-Aware Warning System**  \n"
        "UniNA PICUS Lab · Microsoft LINGUA  \n"
        "PARLA CHIARO Open Call 2025"
    )

    risk = st.session_state.get("last_risk")
    if risk:
        color = _RISK_COLOR[risk]
        st.sidebar.markdown(
            f'<div style="background:{color}22;border:1px solid {color};'
            f'border-radius:8px;padding:10px;margin:10px 0;">'
            f'<span style="color:{color};font-weight:700;">ULTIMO TRIAGE</span><br>'
            f'<span style="color:{color};font-size:15px;">{_RISK_LABEL[risk]}</span><br>'
            f'<span style="color:#666;font-size:11px;">'
            f'U_pipeline = {st.session_state.get("last_u_pipeline",0):.3f}</span></div>',
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    section = st.sidebar.radio("Sezione", ["Inferenza", "Audit Trail", "Analytics"])
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "ASR: WhisperX Large V3 (ctranslate2 CPU int8)  \n"
        "LLM: Mistral 7B v0.3 via Ollama (T=0)  \n"
        "Embed: SBERT multilingual-mpnet-base-v2  \n"
        "UQ: 1D Markov Spettrale Laplace · matrice 6×6 asimmetrica  \n"
        "DB: MongoDB · daws.recordings + daws.inferences"
    )
    return section


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _inject_tooltip_css()
    section = _render_sidebar()
    if section == "Inferenza":
        _section_inferenza()
    elif section == "Audit Trail":
        _section_audit()
    else:
        _section_analytics()


if __name__ == "__main__":
    main()
