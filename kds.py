"""
ELEKTRİKLİ ARACA GEÇİŞ SÜRECİ İÇİN KARAR DESTEK SİSTEMİ
IPCC Tier 2 Metodolojisi & TOPSIS Destekli Senaryo Analizi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EV Geçiş Karar Destek Sistemi",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 2px solid #1e9e6b; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSlider label {
    color: #8b949e !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #1e9e6b !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important; letter-spacing: 0.1em;
}
.stApp { background: #f7f8fc; }
.hero-block {
    background: linear-gradient(135deg, #0d1117 60%, #1a2a1e 100%);
    color: #e6edf3; padding: 2.2rem 2.5rem; border-radius: 12px;
    margin-bottom: 1.5rem; border-left: 5px solid #1e9e6b;
    position: relative; overflow: hidden;
}
.hero-block h1 {
    font-family: 'IBM Plex Mono', monospace; font-size: 27px;
    margin: 0 0 10px 0; color: #1e9e6b; letter-spacing: 0.04em;
}
.hero-block p { font-size: 18px; color: #8b949e; margin: 0; line-height: 1.6; }
.metric-card {
    flex: 1; min-width: 160px; background: #fff; border-radius: 10px;
    padding: 1rem 1.2rem; border: 1px solid #e1e4e8;
    border-top: 3px solid var(--accent, #1e9e6b); box-shadow: 0 1px 4px #0001;
}
.metric-card .val {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem;
    font-weight: 600; color: #0d1117; line-height: 1.1;
}
.metric-card .lbl { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6e7781; margin-top: 4px; }
.winner-box {
    background: linear-gradient(135deg, #1a2a1e, #0d1117);
    border: 1px solid #1e9e6b; border-radius: 10px; padding: 1.4rem 1.8rem;
    color: #e6edf3; font-family: 'IBM Plex Mono', monospace; margin-top: 1rem;
}
.winner-box .wlbl { color: #1e9e6b; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; }
.winner-box .wval { font-size: 1.2rem; font-weight: 600; margin-top: 4px; }
.topsis-card {
    background: #fff; border-radius: 12px; padding: 1.3rem 1.6rem;
    border: 1px solid #e1e4e8; border-left: 4px solid #8B5CF6;
    margin-bottom: 0.9rem; box-shadow: 0 2px 6px #0001;
}
.topsis-card h4 {
    margin: 0 0 0.4rem 0; font-size: 0.8rem; color: #8B5CF6;
    text-transform: uppercase; letter-spacing: 0.1em; font-family: 'IBM Plex Mono', monospace;
}
.topsis-score { font-family: 'IBM Plex Mono', monospace; font-size: 2rem; font-weight: 700; color: #0d1117; }
.topsis-rank {
    display: inline-block; background: #8B5CF6; color: white; border-radius: 50%;
    width: 28px; height: 28px; line-height: 28px; text-align: center;
    font-weight: 700; font-size: 0.9rem; margin-right: 8px; font-family: 'IBM Plex Mono', monospace;
}
.stButton > button {
    background: #1e9e6b !important; color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important; padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.04em; transition: all 0.2s;
}
.stButton > button:hover {
    background: #17845a !important; transform: translateY(-1px);
    box-shadow: 0 4px 12px #1e9e6b44 !important;
}
hr { border-color: #e1e4e8 !important; margin: 1.5rem 0 !important; }
.info-box {
    background: #ddf4e844; border-left: 4px solid #1e9e6b;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    font-size: 0.85rem; color: #1a3a2a; margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─── SABİT DEĞERLER ───
EF_CO2_DIZEL  = 2.690
EF_CH4_OTOBUS = 3.9
EF_N2O_OTOBUS = 3.9
EF_CH4_MINI   = 3.9
EF_N2O_MINI   = 3.9
EF_GRID       = 0.43
ETA_SARJ      = 0.90
E_OTOBUS_EV   = 0.18
E_MINI_EV     = 0.12
TUK_OTOBUS    = 0.33
TUK_MINI      = 0.12
GWP_CH4 = 28
GWP_N2O = 265

RENK = {"MD":"#555555","S1":"#2166AC","S2":"#F4A100","S3":"#1B7837"}
ETIKET = {
    "MD": "Mevcut Durum (Tam Dizel)",
    "S1": "Senaryo 1 – 1/3 EV Geçişi",
    "S2": "Senaryo 2 – 2/3 EV Geçişi",
    "S3": "Senaryo 3 – Tam EV Geçişi",
}

ANALIZ_YILI = 10

# ─── BAŞLIK ───
st.markdown("""
<div class="hero-block">
  <h1>📈 ELEKTRİKLİ ARACA GEÇİŞ KARAR DESTEK SİSTEMİ</h1>
  <p>
    Bu uygulama, dizel araç filolarının elektrikli araçlara geçiş sürecinde emisyon ve maliyet
    parametrelerine dayalı senaryo analizleri gerçekleştirerek geçiş kararlarının optimizasyonunu
    desteklemektedir. Karar yöntemi: <b>TOPSIS</b> (3 Kriter).
  </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("## 📋 FİLO BİLGİLERİ")
    st.markdown("### MEVCUT FİLO")
    n_otobus  = st.number_input("Dizel Otobüs Sayısı", min_value=1, value=20, step=1)
    n_mini    = st.number_input("Dizel Minibüs Sayısı", min_value=0, value=10, step=1)

    st.markdown("### ARAÇ FİYATLARI (TL)")
    fiyat_otobus_ev = st.number_input("Elektrikli Otobüs Birim Fiyatı", min_value=1.0, value=8_000_000.0, step=100_000.0, format="%.0f")
    fiyat_mini_ev   = st.number_input("Elektrikli Minibüs Birim Fiyatı", min_value=0.0, value=3_500_000.0, step=100_000.0, format="%.0f")

    st.markdown("### BAKIM MALİYETLERİ (TL/ARAÇ/YIL)")
    bak_otobus_d = st.number_input("Dizel Otobüs Bakım", min_value=0.0, value=150_000.0, step=10_000.0, format="%.0f")
    bak_mini_d   = st.number_input("Dizel Minibüs Bakım", min_value=0.0, value=80_000.0, step=10_000.0, format="%.0f")
    bak_otobus_e = st.number_input("EV Otobüs Bakım", min_value=0.0, value=60_000.0, step=10_000.0, format="%.0f")
    bak_mini_e   = st.number_input("EV Minibüs Bakım", min_value=0.0, value=35_000.0, step=10_000.0, format="%.0f")

    st.markdown("### YAKIT / ENERJİ FİYATLARI")
    dizel_fiyat    = st.number_input("Dizel Fiyatı (TL/L)", min_value=0.0, value=45.0, step=1.0)
    elektrik_fiyat = st.number_input("Elektrik Fiyatı (TL/kWh)", min_value=0.0, value=4.5, step=0.1)

    st.markdown("### YILLIK KİLOMETRELER")
    km_otobus = st.number_input("Otobüs Filosu Toplam Yıllık km", min_value=1.0, value=1_500_000.0, step=10_000.0, format="%.0f")
    km_mini   = st.number_input("Minibüs Filosu Toplam Yıllık km", min_value=0.0, value=600_000.0, step=10_000.0, format="%.0f")

    st.markdown("### ENFLASYON & ÖDEME")
    tufe_yuzde = st.number_input("Yıllık TÜFE Oranı (%)", min_value=0.0, value=30.0, step=1.0)
    tufe_orani = tufe_yuzde / 100.0
    odeme_plani = st.radio("Ödeme Planı", options=[1,2],
        format_func=lambda x: "Sabit Yıllık Ödeme" if x==1 else f"TÜİK Zam Bazlı (%{tufe_yuzde:.0f}/yıl)", index=0)
    odeme_plani_adi = "Sabit Ödeme Planı" if odeme_plani==1 else f"TÜİK Zam Oranı Bazlı Plan (%{tufe_yuzde:.0f}/yıl)"

    st.markdown("### 🎯 TOPSIS KRİTER AĞIRLIKLARI")
    st.caption("Emisyon + Maliyet ≤ 1.0 olmalı; Yatırım otomatik hesaplanır.")
    w_emisyon = st.slider("Emisyon Ağırlığı",  0.0, 1.0, 0.40, 0.05)
    w_maliyet = st.slider("Maliyet Ağırlığı",  0.0, 1.0 - w_emisyon, 0.35, 0.05)
    w_yatirim = round(max(0.0, 1.0 - w_emisyon - w_maliyet), 4)
    st.markdown(f"**Yatırım Ağırlığı (otomatik):** `{w_yatirim:.2f}`")

    hesapla_btn = st.button("🔍 ANALİZİ ÇALIŞTIR", use_container_width=True)


# ─── TOPSIS FONKSİYONU ───
def topsis(em_s1, em_s2, em_s3, df_s1, df_s2, df_s3, yat1, yat2, yat3, we, wm, wy):
    M = np.array([
        [em_s1["CO2e_ton"], df_s1["toplam"].sum(), yat1],
        [em_s2["CO2e_ton"], df_s2["toplam"].sum(), yat2],
        [em_s3["CO2e_ton"], df_s3["toplam"].sum(), yat3],
    ], dtype=float)

    denom = np.sqrt((M**2).sum(axis=0))
    denom[denom == 0] = 1e-12
    R = M / denom

    W = np.array([we, wm, wy])
    V = R * W

    PIS = V.min(axis=0)
    NIS = V.max(axis=0)

    d_pos = np.sqrt(((V - PIS)**2).sum(axis=1))
    d_neg = np.sqrt(((V - NIS)**2).sum(axis=1))

    C = d_neg / (d_pos + d_neg + 1e-12)
    en_iyi = ["S1","S2","S3"][int(np.argmax(C))]

    return {"M": M, "R": R, "V": V, "PIS": PIS, "NIS": NIS,
            "d_pos": d_pos, "d_neg": d_neg, "C": C, "en_iyi": en_iyi}


# ─── ANA HESAPLAMA ───
def run_analysis(we, wm, wy):
    md = dict(otobus_d=n_otobus, otobus_e=0, mini_d=n_mini, mini_e=0)
    s1 = dict(otobus_d=n_otobus*(2/3), otobus_e=n_otobus/3, mini_d=n_mini*(2/3), mini_e=n_mini/3)
    s2 = dict(otobus_d=n_otobus/3, otobus_e=n_otobus*(2/3), mini_d=n_mini/3, mini_e=n_mini*(2/3))
    s3 = dict(otobus_d=0, otobus_e=n_otobus, mini_d=0, mini_e=n_mini)

    def em(sc, km_o, km_m):
        o_km_d  = (sc["otobus_d"] / max(n_otobus,1)) * km_o
        o_km_e  = (sc["otobus_e"] / max(n_otobus,1)) * km_o
        m_km_d  = (sc["mini_d"] / max(n_mini,1)) * km_m if n_mini>0 else 0
        m_km_e  = (sc["mini_e"] / max(n_mini,1)) * km_m if n_mini>0 else 0
        co2  = o_km_d*TUK_OTOBUS*EF_CO2_DIZEL + o_km_e*(E_OTOBUS_EV/ETA_SARJ)*EF_GRID
        co2 += m_km_d*TUK_MINI*EF_CO2_DIZEL   + m_km_e*(E_MINI_EV/ETA_SARJ)*EF_GRID
        ch4  = (o_km_d*EF_CH4_OTOBUS + m_km_d*EF_CH4_MINI) / 1e6
        n2o  = (o_km_d*EF_N2O_OTOBUS + m_km_d*EF_N2O_MINI) / 1e6
        co2e = (co2 + ch4*GWP_CH4 + n2o*GWP_N2O) / 1000
        return {"CO2_kg": co2, "CH4_kg": ch4, "N2O_kg": n2o, "CO2e_ton": co2e}

    em_md = em(md, km_otobus, km_mini)
    em_s1 = em(s1, km_otobus, km_mini)
    em_s2 = em(s2, km_otobus, km_mini)
    em_s3 = em(s3, km_otobus, km_mini)

    def maliyet(sc, n_ev_o, n_ev_m, f_o, f_m, tufe, yil, plan):
        ay_yak_d = (sc["otobus_d"]*(km_otobus/max(n_otobus,1))*TUK_OTOBUS*dizel_fiyat +
                    sc["mini_d"]*(km_mini/max(n_mini,1) if n_mini>0 else 0)*TUK_MINI*dizel_fiyat) / 12
        ay_yak_e = (sc["otobus_e"]*(km_otobus/max(n_otobus,1))*(E_OTOBUS_EV/ETA_SARJ)*elektrik_fiyat +
                    sc["mini_e"]*(km_mini/max(n_mini,1) if n_mini>0 else 0)*(E_MINI_EV/ETA_SARJ)*elektrik_fiyat) / 12
        ay_bak = (sc["otobus_d"]*bak_otobus_d + sc["mini_d"]*bak_mini_d +
                  sc["otobus_e"]*bak_otobus_e + sc["mini_e"]*bak_mini_e) / 12
        yat = n_ev_o*f_o + n_ev_m*f_m
        taksit_sabit = yat/(yil*12) if plan==1 and yat>0 else 0
        if plan==2 and tufe>0:
            carpan = sum((1+tufe)**t for t in range(yil))
            tst = yat/carpan if carpan>0 else 0
        else:
            tst = yat/yil if plan==2 and yat>0 else 0

        rows = []
        for ay in range(1, yil*12+1):
            yn = (ay-1)//12
            yc = (1+tufe)**yn
            yak  = (ay_yak_d + ay_yak_e)*yc
            bak  = ay_bak*yc
            taks = (taksit_sabit if plan==1 else ((tst/12)*yc if yat>0 else 0))
            rows.append({"ay":ay,"yil":yn+1,"yakıt":yak,"bakım":bak,"taksit":taks,"toplam":yak+bak+taks})
        return pd.DataFrame(rows)

    df_md = maliyet(md, 0, 0, 0, 0, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s1 = maliyet(s1, n_otobus/3, n_mini/3, fiyat_otobus_ev, fiyat_mini_ev, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s2 = maliyet(s2, n_otobus*(2/3), n_mini*(2/3), fiyat_otobus_ev, fiyat_mini_ev, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s3 = maliyet(s3, n_otobus, n_mini, fiyat_otobus_ev, fiyat_mini_ev, tufe_orani, ANALIZ_YILI, odeme_plani)

    yat1 = (n_otobus/3)*fiyat_otobus_ev + (n_mini/3)*fiyat_mini_ev
    yat2 = (n_otobus*(2/3))*fiyat_otobus_ev + (n_mini*(2/3))*fiyat_mini_ev
    yat3 = n_otobus*fiyat_otobus_ev + n_mini*fiyat_mini_ev

    t = topsis(em_s1, em_s2, em_s3, df_s1, df_s2, df_s3, yat1, yat2, yat3, we, wm, wy)

    return {
        "em_md":em_md,"em_s1":em_s1,"em_s2":em_s2,"em_s3":em_s3,
        "df_md":df_md,"df_s1":df_s1,"df_s2":df_s2,"df_s3":df_s3,
        "yat1":yat1,"yat2":yat2,"yat3":yat3,
        "topsis":t, "en_iyi":t["en_iyi"],
    }


# ─── SESSION STATE ───
if "results" not in st.session_state:
    st.session_state["results"] = None

if hesapla_btn:
    with st.spinner("TOPSIS analizi çalışıyor…"):
        st.session_state["results"] = run_analysis(w_emisyon, w_maliyet, w_yatirim)

res = st.session_state["results"]

# ─── SONUÇSUZ EKRAN ───
if res is None:
    st.markdown("""
    <div class="info-box">
    ℹ️ Sol panelden filo bilgilerini ve parametrelerinizi girdikten sonra <b>ANALİZİ ÇALIŞTIR</b> butonuna tıklayın.
    </div>""", unsafe_allow_html=True)

# ─── SONUÇLU EKRAN ───
else:
    em_md = res["em_md"]
    em_s1 = res["em_s1"]
    em_s2 = res["em_s2"]
    em_s3 = res["em_s3"]
    df_md = res["df_md"]
    df_s1 = res["df_s1"]
    df_s2 = res["df_s2"]
    df_s3 = res["df_s3"]
    t  = res["topsis"]
    C  = t["C"]
    en_iyi = res["en_iyi"]

    # Özet metrik kartları
    az1 = (1 - em_s1["CO2e_ton"]/em_md["CO2e_ton"])*100 if em_md["CO2e_ton"]>0 else 0
    az2 = (1 - em_s2["CO2e_ton"]/em_md["CO2e_ton"])*100 if em_md["CO2e_ton"]>0 else 0
    az3 = (1 - em_s3["CO2e_ton"]/em_md["CO2e_ton"])*100 if em_md["CO2e_ton"]>0 else 0
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card" style="--accent:#555555"><div class="lbl">MD Yıllık Emisyon</div><div class="val">{em_md["CO2e_ton"]:,.0f}</div><div class="lbl">ton CO₂e/yıl</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card" style="--accent:#2166AC"><div class="lbl">S1 Emisyon Azalması</div><div class="val">▼{az1:.1f}%</div><div class="lbl">Mevcut duruma kıyasla</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card" style="--accent:#F4A100"><div class="lbl">S2 Emisyon Azalması</div><div class="val">▼{az2:.1f}%</div><div class="lbl">Mevcut duruma kıyasla</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card" style="--accent:#1B7837"><div class="lbl">S3 Emisyon Azalması</div><div class="val">▼{az3:.1f}%</div><div class="lbl">Mevcut duruma kıyasla</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab_emisyon, tab_maliyet, tab_topsis, tab_tablo = st.tabs([
         "♻️ EMİSYON ANALİZİ", "💹 MALİYET ANALİZİ", "🎯 TOPSIS ANALİZİ", "📊 DETAY TABLOLAR"
    ])

    # Kazanan kutusu
    st.markdown(f"""
    <div class="winner-box">
        <div class="wlbl">🏆 TOPSIS – ÖNERİLEN OPTİMAL SENARYO</div>
        <div class="wval">{ETIKET[en_iyi]}</div>
        <div style="font-size:0.85rem;color:#8b949e;margin-top:6px;">
            Emisyon (%{w_emisyon*100:.0f}) · Toplam Maliyet (%{w_maliyet*100:.0f}) · Yatırım Maliyeti (%{w_yatirim*100:.0f})
            kriterleri TOPSIS yöntemiyle ağırlıklandırılmıştır.
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── EMİSYON SEKMESİ ─────────────────────────────────────────
    with tab_emisyon:
        st.subheader("IPCC Tier 2 – Senaryo Bazlı Yıllık Emisyon Karşılaştırması")
        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['axes.unicode_minus'] = False
        em_tum = [em_md,em_s1,em_s2,em_s3]
        renkler4 = [RENK[k] for k in ["MD","S1","S2","S3"]]
        fig, axes = plt.subplots(2, 2, figsize=(13,8))
        fig.suptitle("IPCC Tier 2 – Senaryo Bazlı Yıllık Emisyon Karşılaştırması", fontsize=12, fontweight="bold")
        for ax, key, birim, fmt, div in [
            (axes[0,0],"CO2_kg","CO₂ (ton/yıl)",",.0f",1000),
            (axes[0,1],"CH4_kg","CH₄ (kg/yıl)",",.3f",1),
            (axes[1,0],"N2O_kg","N₂O (kg/yıl)",",.3f",1),
            (axes[1,1],"CO2e_ton","CO₂e (ton/yıl)",",.1f",1),
        ]:
            vals = [e[key]/div for e in em_tum]
            bars = ax.bar(["MD","S1","S2","S3"], vals, color=renkler4, width=0.5, edgecolor="white", linewidth=1.2)
            ax.set_title(key.replace("_kg","").replace("_ton","").upper()+" Emisyonu", fontweight="bold")
            ax.set_ylabel(birim)
            mx = max(vals) if max(vals)>0 else 1
            for bar,v in zip(bars,vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+mx*0.01, format(v,fmt), ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_ylim(0, mx*1.18)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### Emisyon Karşılaştırma Tablosu")
        def azalma(v, ref): return f"▼%{(1-v/ref)*100:.1f}" if ref>0 else "-"
        tablo = {
            "Gösterge": ["CO₂ (ton/yıl)","CH₄ (kg/yıl)","N₂O (kg/yıl)","CO₂e (ton/yıl)"],
            "MD": [f"{em_md['CO2_kg']/1000:,.1f}",f"{em_md['CH4_kg']:,.3f}",f"{em_md['N2O_kg']:,.3f}",f"{em_md['CO2e_ton']:,.2f}"],
            "S1": [f"{em_s1['CO2_kg']/1000:,.1f}",f"{em_s1['CH4_kg']:,.3f}",f"{em_s1['N2O_kg']:,.3f}",f"{em_s1['CO2e_ton']:,.2f}"],
            "S1▼": [azalma(em_s1['CO2_kg'],em_md['CO2_kg']),azalma(em_s1['CH4_kg'],em_md['CH4_kg']),azalma(em_s1['N2O_kg'],em_md['N2O_kg']),azalma(em_s1['CO2e_ton'],em_md['CO2e_ton'])],
            "S2": [f"{em_s2['CO2_kg']/1000:,.1f}",f"{em_s2['CH4_kg']:,.3f}",f"{em_s2['N2O_kg']:,.3f}",f"{em_s2['CO2e_ton']:,.2f}"],
            "S2▼": [azalma(em_s2['CO2_kg'],em_md['CO2_kg']),azalma(em_s2['CH4_kg'],em_md['CH4_kg']),azalma(em_s2['N2O_kg'],em_md['N2O_kg']),azalma(em_s2['CO2e_ton'],em_md['CO2e_ton'])],
            "S3": [f"{em_s3['CO2_kg']/1000:,.1f}",f"{em_s3['CH4_kg']:,.3f}",f"{em_s3['N2O_kg']:,.3f}",f"{em_s3['CO2e_ton']:,.2f}"],
            "S3▼": [azalma(em_s3['CO2_kg'],em_md['CO2_kg']),azalma(em_s3['CH4_kg'],em_md['CH4_kg']),azalma(em_s3['N2O_kg'],em_md['N2O_kg']),azalma(em_s3['CO2e_ton'],em_md['CO2e_ton'])],
        }
        st.dataframe(pd.DataFrame(tablo), use_container_width=True, hide_index=True)
        st.info("AZALMA DEĞERLERİ MEVCUT DURUMA GÖRE KIYASLANMIŞTIR.")

 # ──────────────────────────────────────────
    #  MALİYET SEKMESİ
    # ──────────────────────────────────────────
    with tab_maliyet:
        st.subheader("📈 Senaryo Bazlı Başabaş ve Kümülatif Kâr Analizi")

        # Grafik Hazırlığı
        fig_be, ax_be = plt.subplots(figsize=(13, 6))

        # Referans dizel maliyet serileri
        cum_md_yakit_bakim = (df_md["yakıt"] + df_md["bakım"]).cumsum()

        # Her senaryo için kümülatif kâr hesaplama ve çizdirme
        senaryo_listesi = [("S1", df_s1), ("S2", df_s2), ("S3", df_s3)]

        for kod, df_sc in senaryo_listesi:
            # Senaryonun toplam maliyet serisi
            cum_sc_toplam = (df_sc["yakıt"] + df_sc["bakım"] + df_sc["taksit"]).cumsum()

            # Kümülatif Kâr = Dizel Maliyeti - EV Maliyeti (Milyon TL cinsinden)
            cum_kar = (cum_md_yakit_bakim - cum_sc_toplam) / 1e6

            # Grafiğe çizgi ekleme
            ax_be.plot(df_sc["ay"], cum_kar, color=RENK[kod], linewidth=2.5, label=f"{ETIKET[kod]} Kâr Eğrisi")

            # Başabaş noktasını bulma (Kârın ilk kez >= 0 olduğu ay)
            passed_zero = df_sc["ay"][cum_kar >= 1]

            if not passed_zero.empty:
                be_ay = passed_zero.iloc[0]
                be_kar = cum_kar.iloc[be_ay - 1]
                be_yil = be_ay / 12

                # Başabaş noktasını kırmızı halka ile işaretle
                ax_be.plot(be_ay, be_kar, marker="o", color="red", markersize=8, zorder=5)

                # Metinsel Etiketleme
                offset = 5 if kod == "S3" else (-15 if kod == "S1" else -5)
                ax_be.annotate(f" Amorti: {be_ay}. Ay\n ({be_yil:.1f} Yıl)",
                               xy=(be_ay, be_kar),
                               xytext=(be_ay + 3, be_kar + offset),
                               color=RENK[kod], fontsize=8, fontweight="bold",
                               arrowprops=dict(arrowstyle="->", color=RENK[kod], alpha=0.6),
                               bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
            else:
                ax_be.text(df_sc["ay"].iloc[-1], cum_kar.iloc[-1], " Amorti Edilemedi",
                           color=RENK[kod], fontsize=8, linestyle="--")

        # Kar/Zarar Eşiği (0 Çizgisi)
        ax_be.axhline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.7)
        ax_be.fill_between(df_md["ay"], 0, ax_be.get_ylim()[0], color="#fcd7d7", alpha=0.15, label="Zarar Bölgesi")
        ax_be.fill_between(df_md["ay"], 0, ax_be.get_ylim()[1], color="#ddf4e8", alpha=0.15, label="Net Kâr Bölgesi")

        # Grafik Tasarımı ve Düzenlemeler
        ax_be.set_title(f"Zamana Bağlı Kümülatif Kâr ve Başabaş Noktaları ({ANALIZ_YILI} Yıllık Süreç)", fontweight="bold", fontsize=12)
        ax_be.set_xlabel("Zaman Ekseni (Ay)", fontweight="bold")
        ax_be.set_ylabel("Kümülatif Net Kâr / Tasarruf (Milyon TL)", fontweight="bold")
        ax_be.legend(loc="upper left", fontsize=9)
        ax_be.xaxis.set_major_locator(mticker.MultipleLocator(12))
        ax_be.grid(True, which='both', linestyle=':', alpha=0.5)

        # Yılları dikey çizgilerle ayırma
        for y in range(1, ANALIZ_YILI + 1):
            ax_be.axvline(y * 12, color="gray", lw=0.4, alpha=0.25, linestyle="-")

        plt.tight_layout()
        st.pyplot(fig_be)
        plt.close(fig_be)

        st.markdown("---")
        st.subheader(f"Senaryo Bazlı Aylık Maliyet Analizi – {ANALIZ_YILI} Yıl Detayları")
        st.caption(f"Ödeme Planı: {odeme_plani_adi} | TÜFE: %{tufe_yuzde:.1f}")

        for df_, kod in [(df_s1,"S1"), (df_s2,"S2"), (df_s3,"S3")]:
            with st.expander(f"📊 {ETIKET[kod]}", expanded=(kod=="S2")):
                col_g, col_t = st.columns([2, 1])
                with col_g:
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    renk = RENK[kod]
                    ax2.fill_between(df_["ay"], df_["yakıt"]/1e6, alpha=0.4, color=renk, label="Yakıt")
                    ax2.fill_between(df_["ay"], (df_["yakıt"]+df_["bakım"])/1e6,
                                     df_["yakıt"]/1e6, alpha=0.4, color="gray", label="Bakım")
                    ax2.fill_between(df_["ay"], df_["toplam"]/1e6,
                                     (df_["yakıt"]+df_["bakım"])/1e6,
                                     alpha=0.4, color="orange", label="Araç Taksiti")
                    ax2.plot(df_["ay"], df_["toplam"]/1e6, color=renk, linewidth=2, label="Toplam")
                    ax2.set_title(f"{ETIKET[kod]}: Aylık Maliyet Bileşenleri", fontweight="bold")
                    ax2.set_xlabel("Ay"); ax2.set_ylabel("Milyon TL")
                    ax2.legend(loc="upper left", fontsize=8)
                    ax2.xaxis.set_major_locator(mticker.MultipleLocator(12))
                    for y in range(1, ANALIZ_YILI+1):
                        ax2.axvline(y*12, color="gray", lw=0.5, alpha=0.35, linestyle="--")
                    plt.tight_layout()
                    st.pyplot(fig2); plt.close(fig2)

                with col_t:
                    yillik_df = df_.groupby("yil")[["yakıt", "bakım", "taksit", "toplam"]].mean().reset_index()
                    yillik_df.columns = ["Yıl", "Yakıt (TL)", "Bakım (TL)", "Taksit (TL)", "Toplam (TL)"]
                    for c in ["Yakıt (TL)", "Bakım (TL)", "Taksit (TL)", "Toplam (TL)"]:
                        yillik_df[c] = yillik_df[c].map(lambda x: f"{x:,.0f}")
                    yillik_df["Yıl"] = yillik_df["Yıl"].astype(int)
                    st.dataframe(yillik_df, use_container_width=True, hide_index=True,
                                 height=min(40 + ANALIZ_YILI * 35, 500))


    # ── TOPSIS SEKMESİ ──────────────────────────────────────────
    with tab_topsis:
        st.subheader("TOPSIS – 3 Kriterli Çok Amaçlı Karar Analizi")
        st.caption(f"Kriterler: CO₂e Emisyonu · Toplam Maliyet · Yatırım Maliyeti  |  Ağırlıklar: {w_emisyon:.2f} / {w_maliyet:.2f} / {w_yatirim:.2f}")

        rank_colors = {1:"#1e9e6b", 2:"#F4A100", 3:"#e05c5c"}
        sen_info = [("S1","1/3 EV Geçişi",RENK["S1"]),("S2","2/3 EV Geçişi",RENK["S2"]),("S3","Tam EV Geçişi",RENK["S3"])]
        ranks_order = np.argsort(-C)

        col1, col2, col3 = st.columns(3)
        for col, (idx,(kod,ad,renk)) in zip([col1,col2,col3], enumerate(sen_info)):
            rank = int(np.where(ranks_order==idx)[0][0]) + 1
            rk_c = rank_colors.get(rank,"#888")
            with col:
                st.markdown(f"""
                <div class="topsis-card" style="border-left-color:{renk};">
                    <h4>{kod} – {ad}</h4>
                    <div style="display:flex;align-items:center;gap:10px;margin:8px 0;">
                        <span class="topsis-rank" style="background:{rk_c};">#{rank}</span>
                        <span class="topsis-score" style="color:{renk};">{C[idx]:.4f}</span>
                    </div>
                    <div style="font-size:0.75rem;color:#6e7781;margin-top:4px;">C* Yakınlık Skoru — Yüksek = Daha İyi</div>
                    <div style="background:#f0f0f0;border-radius:6px;height:8px;margin-top:10px;overflow:hidden;">
                        <div style="background:{renk};width:{C[idx]*100:.1f}%;height:100%;border-radius:6px;"></div>
                    </div>
                    <div style="font-size:0.7rem;color:#aaa;margin-top:3px;">d⁺={t['d_pos'][idx]:.4f} &nbsp;|&nbsp; d⁻={t['d_neg'][idx]:.4f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cg1, cg2 = st.columns(2)
        with cg1:
            fig, ax = plt.subplots(figsize=(6,4))
            labels = ["S1\n1/3 EV","S2\n2/3 EV","S3\nTam EV"]
            renkler = [RENK["S1"],RENK["S2"],RENK["S3"]]
            bars = ax.bar(labels, C, color=renkler, width=0.45, edgecolor="white", linewidth=1.5)
            for bar,cv in zip(bars,C):
                ax.text(bar.get_x()+bar.get_width()/2, cv+0.005, f"{cv:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.set_title("TOPSIS Yakınlık Skorları (C*)", fontweight="bold", fontsize=11)
            ax.set_ylabel("C* Skoru (0–1)")
            ax.set_ylim(0, min(1.05, max(C)*1.3))
            ax.axhline(0.5, color="gray", lw=1, linestyle="--", alpha=0.5)
            ax.text(2.45, 0.51, "0.5", color="gray", fontsize=8)
            ax.grid(True, axis='y', linestyle=':', alpha=0.4)
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with cg2:
            fig2, ax2 = plt.subplots(figsize=(6,4))
            x = np.arange(3)
            w = 0.35
            ax2.bar(x-w/2, t["d_pos"], w, label="d⁺ (Pozitif İdeal Uzaklığı)", color=renkler, alpha=0.85, edgecolor="white")
            ax2.bar(x+w/2, t["d_neg"], w, label="d⁻ (Negatif İdeal Uzaklığı)", color=renkler, alpha=0.4, edgecolor="white", hatch="//")
            ax2.set_xticks(x)
            ax2.set_xticklabels(["S1","S2","S3"])
            ax2.set_title("İdeal Çözüme Öklid Uzaklıkları", fontweight="bold", fontsize=11)
            ax2.set_ylabel("Normalize Uzaklık")
            ax2.legend(fontsize=8)
            ax2.grid(True, axis='y', linestyle=':', alpha=0.4)
            ax2.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("---")
        st.markdown("#### 📋 TOPSIS Adım Adım Hesaplama Tabloları")
        krit = ["CO₂e Emisyonu (ton/yıl)", f"Toplam Maliyet ({ANALIZ_YILI}yıl, TL)", "Yatırım Maliyeti (TL)"]

        with st.expander("1️⃣ Ham Karar Matrisi", expanded=True):
            st.dataframe(pd.DataFrame(t["M"], index=["S1","S2","S3"], columns=krit).style.format("{:,.2f}"), use_container_width=True)

        with st.expander("2️⃣ Normalize Edilmiş Matris (R)"):
            st.dataframe(pd.DataFrame(t["R"], index=["S1","S2","S3"], columns=krit).style.format("{:.6f}"), use_container_width=True)

        with st.expander("3️⃣ Ağırlıklı Normalize Matris (V)"):
            st.caption(f"Ağırlıklar: Emisyon={w_emisyon:.2f}, Maliyet={w_maliyet:.2f}, Yatırım={w_yatirim:.2f}")
            st.dataframe(pd.DataFrame(t["V"], index=["S1","S2","S3"], columns=krit).style.format("{:.6f}"), use_container_width=True)

        with st.expander("4️⃣ Pozitif (A⁺) ve Negatif (A⁻) İdeal Çözümler"):
            df_ideal = pd.DataFrame([t["PIS"],t["NIS"]], index=["A⁺ Pozitif İdeal (en küçük)","A⁻ Negatif İdeal (en büyük)"], columns=krit)
            st.dataframe(df_ideal.style.format("{:.6f}"), use_container_width=True)

        with st.expander("5️⃣ Uzaklıklar ve C* Nihai Skoru", expanded=True):
            df_skor = pd.DataFrame({
                "Senaryo": ["S1 – 1/3 EV","S2 – 2/3 EV","S3 – Tam EV"],
                "d⁺ (Pozitif İdeal)": t["d_pos"],
                "d⁻ (Negatif İdeal)": t["d_neg"],
                "C* Yakınlık Skoru": C,
                "Sıralama": [int(np.where(ranks_order==i)[0][0])+1 for i in range(3)],
            })
            st.dataframe(df_skor.style.format({
                "d⁺ (Pozitif İdeal)":"{:.6f}","d⁻ (Negatif İdeal)":"{:.6f}","C* Yakınlık Skoru":"{:.4f}"
            }).background_gradient(subset=["C* Yakınlık Skoru"], cmap="Greens"),
            use_container_width=True, hide_index=True)

    # ── DETAY TABLOLAR ──────────────────────────────────────────
    with tab_tablo:
        st.subheader("Senaryolara Ait Aylık Ham Veri Çıktıları")
        sec = st.selectbox("Görüntülemek İstediğiniz Senaryoyu Seçin:", ["S1","S2","S3"])
        gdf = {"S1":df_s1,"S2":df_s2,"S3":df_s3}[sec].copy()
        gdf.columns = ["Ay","Yıl","Yakıt Maliyeti (TL)","Bakım Maliyeti (TL)","Yatırım Taksiti (TL)","Toplam Aylık Maliyet (TL)"]
        st.dataframe(gdf.style.format({
            "Yakıt Maliyeti (TL)":"{:,.2f}","Bakım Maliyeti (TL)":"{:,.2f}",
            "Yatırım Taksiti (TL)":"{:,.2f}","Toplam Aylık Maliyet (TL)":"{:,.2f}"
        }), use_container_width=True, hide_index=True)
