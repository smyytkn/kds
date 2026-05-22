"""
ELEKTRİKLİ ARACA GEÇİŞ SÜRECİ İÇİN KARAR DESTEK SİSTEMİ
IPCC Tier 2 Metodolojisi
Karabük Üniversitesi – Endüstri Mühendisliği Lisans Bitirme Tezi
Özge ÖZBAY & Sümeyye TEKİN
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
section[data-testid="stSidebar"] .stRadio label {
    color: #8b949e !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
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
.hero-block h1 { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem;
    margin: 0 0 0.3rem 0; color: #1e9e6b; letter-spacing: 0.04em; }
.hero-block p { font-size: 0.88rem; color: #8b949e; margin: 0; }
.metric-card { flex: 1; min-width: 160px; background: #fff; border-radius: 10px;
    padding: 1rem 1.2rem; border: 1px solid #e1e4e8;
    border-top: 3px solid var(--accent, #1e9e6b); box-shadow: 0 1px 4px #0001; }
.metric-card .val { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem;
    font-weight: 600; color: #0d1117; line-height: 1.1; }
.metric-card .lbl { font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.08em; color: #6e7781; margin-top: 4px; }
.winner-box { background: linear-gradient(135deg, #1a2a1e, #0d1117);
    border: 1px solid #1e9e6b; border-radius: 10px; padding: 1.4rem 1.8rem;
    color: #e6edf3; font-family: 'IBM Plex Mono', monospace; margin-top: 1rem; }
.winner-box .wlbl { color: #1e9e6b; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; }
.winner-box .wval { font-size: 1.2rem; font-weight: 600; margin-top: 4px; }
.info-box { background: #ddf4e844; border-left: 4px solid #1e9e6b;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; font-size: 0.85rem;
    color: #1a3a2a; margin: 0.8rem 0; }
.stButton > button { background: #1e9e6b !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.88rem !important;
    padding: 0.6rem 1.8rem !important; font-weight: 600 !important; }
.stButton > button:hover { background: #17845a !important; }
</style>
""", unsafe_allow_html=True)

# ─── SABİT DEĞERLER (IPCC Tier 2) ───
EF_CO2_DIZEL   = 2.690
EF_CO2_BENZIN  = 2.350
EF_CH4_OTOBÜS_DIZEL  = 3.9
EF_N2O_OTOBÜS_DIZEL  = 3.9
EF_CH4_MINİBÜS_DIZEL = 3.9
EF_N2O_MINİBÜS_DIZEL = 3.9
EF_GRID      = 0.43
ETA_SARJ     = 0.90
E_OTOBÜS_EV  = 0.18
E_MINİBÜS_EV = 0.12
TUK_OTOBÜS_DIZEL  = 0.33
TUK_MINİBÜS_DIZEL = 0.12
GWP_CH4 = 28
GWP_N2O = 265

RENK = {"MD": "#555555", "S1": "#2166AC", "S2": "#F4A100", "S3": "#1B7837"}
ETIKET = {
    "MD": "Mevcut Durum (Tam Dizel)",
    "S1": "Senaryo 1 – 1/3 EV Geçişi",
    "S2": "Senaryo 2 – 2/3 EV Geçişi",
    "S3": "Senaryo 3 – Tam EV Geçişi",
}

# ─── BAŞLIK ───
st.markdown("""
<div class="hero-block">
  <h1>📈 ELEKTRİKLİ ARACA GEÇİŞ KARAR DESTEK SİSTEMİ</h1>
  <p>IPCC Tier 2 Metodolojisi ile Emisyon & Maliyet Senaryo Analizi</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("## 📋 FİLO BİLGİLERİ")
    st.markdown("### MEVCUT FİLO")
    n_otobüs_mevcut  = st.number_input("Dizel Otobüs Sayısı (adet)", min_value=1, value=20, step=1)
    n_minibüs_mevcut = st.number_input("Dizel Minibüs Sayısı (adet)", min_value=0, value=10, step=1)

    st.markdown("### ARAÇ FİYATLARI (TL)")
    fiyat_otobüs_ev  = st.number_input("Elektrikli Otobüs Birim Fiyatı (TL)", min_value=1.0, value=8_000_000.0, step=100_000.0, format="%.0f")
    fiyat_minibüs_ev = st.number_input("Elektrikli Minibüs Birim Fiyatı (TL)", min_value=0.0, value=3_500_000.0, step=100_000.0, format="%.0f")

    st.markdown("### BAKIM MALİYETLERİ (TL/ARAÇ/YIL)")
    bakim_otobüs_dizel  = st.number_input("Dizel Otobüs Bakım", min_value=0.0, value=150_000.0, step=10_000.0, format="%.0f")
    bakim_minibüs_dizel = st.number_input("Dizel Minibüs Bakım", min_value=0.0, value=80_000.0,  step=10_000.0, format="%.0f")
    bakim_otobüs_ev     = st.number_input("EV Otobüs Bakım",     min_value=0.0, value=60_000.0,  step=10_000.0, format="%.0f")
    bakim_minibüs_ev    = st.number_input("EV Minibüs Bakım",    min_value=0.0, value=35_000.0,  step=10_000.0, format="%.0f")

    st.markdown("### YAKIT / ENERJİ FİYATLARI")
    dizel_fiyat    = st.number_input("Dizel Fiyatı (TL/L)",      min_value=0.0, value=45.0, step=1.0)
    elektrik_fiyat = st.number_input("Elektrik Fiyatı (TL/kWh)", min_value=0.0, value=4.5,  step=0.1)

    st.markdown("### YILLIK KİLOMETRELER")
    km_otobüs_yillik  = st.number_input("Otobüs Filosu Toplam Yıllık km", min_value=1.0, value=1_500_000.0, step=10_000.0, format="%.0f")
    km_minibüs_yillik = st.number_input("Minibüs Filosu Toplam Yıllık km", min_value=0.0, value=600_000.0,   step=10_000.0, format="%.0f")

    st.markdown("### ENFLASYON & ÖDEME")
    tufe_yuzde = st.number_input("Yıllık TÜFE Oranı (%)", min_value=0.0, value=30.0, step=1.0)
    tufe_orani = tufe_yuzde / 100.0

    odeme_plani = st.radio(
        "Ödeme Planı",
        options=[1, 2],
        format_func=lambda x: "Sabit Yıllık Ödeme" if x == 1 else f"TÜİK Zam Bazlı (%{tufe_yuzde:.0f}/yıl)",
        index=0,
    )
    odeme_plani_adi = "Sabit Ödeme Planı" if odeme_plani == 1 else f"TÜİK Zam Oranı Bazlı Plan (%{tufe_yuzde:.0f}/yıl)"

    w_emisyon = 0.5
    w_maliyet = 0.5
    hesapla_btn = st.button("🔍 ANALİZİ ÇALIŞTIR", use_container_width=True)

ANALIZ_YILI = 10  # Her iki ödeme planı için de 10 yıl

# ─── HESAPLAMALAR ───
def run_analysis(w_emisyon, w_maliyet):
    md = dict(otobüs_dizel=n_otobüs_mevcut,          otobüs_ev=0,
              minibüs_dizel=n_minibüs_mevcut,         minibüs_ev=0)
    s1 = dict(otobüs_dizel=n_otobüs_mevcut*(2/3),    otobüs_ev=n_otobüs_mevcut/3,
              minibüs_dizel=n_minibüs_mevcut*(2/3),   minibüs_ev=n_minibüs_mevcut/3)
    s2 = dict(otobüs_dizel=n_otobüs_mevcut/3,        otobüs_ev=n_otobüs_mevcut*(2/3),
              minibüs_dizel=n_minibüs_mevcut/3,       minibüs_ev=n_minibüs_mevcut*(2/3))
    s3 = dict(otobüs_dizel=0,                         otobüs_ev=n_otobüs_mevcut,
              minibüs_dizel=0,                         minibüs_ev=n_minibüs_mevcut)

    def emisyon_hesapla(senaryo, km_oto, km_mini):
        r = {}
        oto_km_d  = (senaryo["otobüs_dizel"] / max(n_otobüs_mevcut, 1)) * km_oto
        co2_od    = oto_km_d * TUK_OTOBÜS_DIZEL * EF_CO2_DIZEL
        ch4_od    = oto_km_d * EF_CH4_OTOBÜS_DIZEL / 1e6
        n2o_od    = oto_km_d * EF_N2O_OTOBÜS_DIZEL / 1e6
        oto_km_ev = (senaryo["otobüs_ev"] / max(n_otobüs_mevcut, 1)) * km_oto
        co2_oev   = oto_km_ev * (E_OTOBÜS_EV / ETA_SARJ) * EF_GRID
        mini_km_d  = (senaryo["minibüs_dizel"] / max(n_minibüs_mevcut, 1)) * km_mini if n_minibüs_mevcut > 0 else 0
        co2_md_    = mini_km_d * TUK_MINİBÜS_DIZEL * EF_CO2_DIZEL
        ch4_md_    = mini_km_d * EF_CH4_MINİBÜS_DIZEL / 1e6
        n2o_md_    = mini_km_d * EF_N2O_MINİBÜS_DIZEL / 1e6
        mini_km_ev = (senaryo["minibüs_ev"] / max(n_minibüs_mevcut, 1)) * km_mini if n_minibüs_mevcut > 0 else 0
        co2_mev    = mini_km_ev * (E_MINİBÜS_EV / ETA_SARJ) * EF_GRID
        r["CO2_kg"]   = co2_od + co2_oev + co2_md_ + co2_mev
        r["CH4_kg"]   = ch4_od + ch4_md_
        r["N2O_kg"]   = n2o_od + n2o_md_
        r["CO2e_ton"] = (r["CO2_kg"] + r["CH4_kg"] * GWP_CH4 + r["N2O_kg"] * GWP_N2O) / 1000
        return r

    em_md = emisyon_hesapla(md, km_otobüs_yillik, km_minibüs_yillik)
    em_s1 = emisyon_hesapla(s1, km_otobüs_yillik, km_minibüs_yillik)
    em_s2 = emisyon_hesapla(s2, km_otobüs_yillik, km_minibüs_yillik)
    em_s3 = emisyon_hesapla(s3, km_otobüs_yillik, km_minibüs_yillik)

    # ── Aylık maliyet serisi üretici ──
    # Her iki plan için:
    #   - MD: yatırım=0, sadece yakıt+bakım (TÜFE ile artar)
    #   - Sx: yakıt+bakım (TÜFE ile artar) + taksit
    #     Sabit plan: taksit = yatırım / (10*12) sabit
    #     TÜFE plan:  taksit_0 * (1+tüfe)^yil
    def maliyet_serileri(senaryo, n_ev_oto, n_ev_mini, fiy_oto_ev, fiy_mini_ev, tufe, yil, odeme_plan):
        # Ay 0 (yıl 0) yakıt+bakım
        yillik_yakıt_dizel = (
            senaryo["otobüs_dizel"]  * (km_otobüs_yillik  / n_otobüs_mevcut  if n_otobüs_mevcut  else 0) * TUK_OTOBÜS_DIZEL  * dizel_fiyat +
            senaryo["minibüs_dizel"] * (km_minibüs_yillik / n_minibüs_mevcut if n_minibüs_mevcut else 0) * TUK_MINİBÜS_DIZEL * dizel_fiyat
        )
        yillik_yakıt_ev = (
            senaryo["otobüs_ev"]  * (km_otobüs_yillik  / n_otobüs_mevcut  if n_otobüs_mevcut  else 0) * (E_OTOBÜS_EV  / ETA_SARJ) * elektrik_fiyat +
            senaryo["minibüs_ev"] * (km_minibüs_yillik / n_minibüs_mevcut if n_minibüs_mevcut else 0) * (E_MINİBÜS_EV / ETA_SARJ) * elektrik_fiyat
        )
        yillik_bakım = (
            senaryo["otobüs_dizel"]  * bakim_otobüs_dizel  +
            senaryo["minibüs_dizel"] * bakim_minibüs_dizel +
            senaryo["otobüs_ev"]     * bakim_otobüs_ev     +
            senaryo["minibüs_ev"]    * bakim_minibüs_ev
        )

        yatirim = n_ev_oto * fiy_oto_ev + n_ev_mini * fiy_mini_ev
        toplam_ay = yil * 12

        if odeme_plan == 1:
            taksit_sabit = yatirim / toplam_ay if toplam_ay > 0 else 0
        else:
            # TÜFE bazlı: t=0 ödeme * sum(1+tüfe)^t for t=0..yil-1 = yatirim
            if tufe > 0:
                carpan_t = sum((1 + tufe) ** t for t in range(yil))
                taksit_0_yillik = yatirim / carpan_t if carpan_t > 0 else 0
            else:
                taksit_0_yillik = yatirim / yil if yil > 0 else 0

        kayitlar = []
        for ay in range(1, toplam_ay + 1):
            yn = (ay - 1) // 12   # 0-indexed yıl
            yc = (1 + tufe) ** yn

            yakıt = (yillik_yakıt_dizel + yillik_yakıt_ev) * yc / 12
            bakım = yillik_bakım * yc / 12

            if odeme_plan == 1:
                taksit = taksit_sabit
            else:
                taksit = (taksit_0_yillik / 12) * yc if yatirim > 0 else 0

            kayitlar.append({
                "ay": ay, "yil": yn + 1,
                "yakıt": yakıt, "bakım": bakım, "taksit": taksit,
                "isletme": yakıt + bakım,
                "toplam": yakıt + bakım + taksit
            })
        return pd.DataFrame(kayitlar), yatirim

    df_md, _    = maliyet_serileri(md, 0,                        0,                        0,              0,              tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s1, yat1 = maliyet_serileri(s1, n_otobüs_mevcut/3,       n_minibüs_mevcut/3,       fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s2, yat2 = maliyet_serileri(s2, n_otobüs_mevcut*(2/3),   n_minibüs_mevcut*(2/3),   fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s3, yat3 = maliyet_serileri(s3, n_otobüs_mevcut,         n_minibüs_mevcut,         fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)

    # AHP
    em_d  = np.array([em_s1["CO2e_ton"], em_s2["CO2e_ton"], em_s3["CO2e_ton"]])
    mal_d = np.array([df_s1["toplam"].sum(), df_s2["toplam"].sum(), df_s3["toplam"].sum()])
    def norm_min(v):
        inv = 1.0 / (v + 1e-12)
        return inv / inv.sum()
    em_norm  = norm_min(em_d)
    mal_norm = norm_min(mal_d)
    ahp      = w_emisyon * em_norm + w_maliyet * mal_norm
    en_iyi   = ["S1", "S2", "S3"][np.argmax(ahp)]

    return {
        "em_md": em_md, "em_s1": em_s1, "em_s2": em_s2, "em_s3": em_s3,
        "df_md": df_md, "df_s1": df_s1, "df_s2": df_s2, "df_s3": df_s3,
        "yat1": yat1,   "yat2": yat2,   "yat3": yat3,
        "em_norm": em_norm, "mal_norm": mal_norm, "ahp": ahp, "en_iyi": en_iyi,
    }

# ─── OTURUM ───
if "results" not in st.session_state:
    st.session_state["results"] = None

if hesapla_btn:
    with st.spinner("Analiz çalışıyor…"):
        st.session_state["results"] = run_analysis(w_emisyon, w_maliyet)

res = st.session_state["results"]

# ─── GÖSTERİM ───
if res is None:
    st.markdown("""
    <div class="info-box">
    ℹ️ Sol panelden filo bilgilerini ve parametrelerinizi girdikten sonra <b>ANALİZİ ÇALIŞTIR</b> butonuna tıklayın.
    </div>
    """, unsafe_allow_html=True)
else:
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.unicode_minus'] = False

    em_md, em_s1, em_s2, em_s3 = res["em_md"], res["em_s1"], res["em_s2"], res["em_s3"]
    df_md, df_s1, df_s2, df_s3 = res["df_md"], res["df_s1"], res["df_s2"], res["df_s3"]
    yat1,  yat2,  yat3         = res["yat1"],  res["yat2"],  res["yat3"]
    ahp, em_norm, mal_norm, en_iyi = res["ahp"], res["em_norm"], res["mal_norm"], res["en_iyi"]

    # ── Özet metrik kartları ──
    s1_azalma = (1 - em_s1["CO2e_ton"] / em_md["CO2e_ton"]) * 100 if em_md["CO2e_ton"] > 0 else 0
    s2_azalma = (1 - em_s2["CO2e_ton"] / em_md["CO2e_ton"]) * 100 if em_md["CO2e_ton"] > 0 else 0
    s3_azalma = (1 - em_s3["CO2e_ton"] / em_md["CO2e_ton"]) * 100 if em_md["CO2e_ton"] > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card" style="--accent:#555555">
          <div class="lbl">Mevcut Durum Yıllık Emisyon</div>
          <div class="val">{em_md['CO2e_ton']:,.0f}</div>
          <div class="lbl">ton CO₂e/yıl</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="--accent:#2166AC">
          <div class="lbl">S1 – 1/3 EV Emisyon Azalması</div>
          <div class="val">▼{s1_azalma:.1f}%</div>
          <div class="lbl">Mevcut Duruma Göre</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="--accent:#F4A100">
          <div class="lbl">S2 – 2/3 EV Emisyon Azalması</div>
          <div class="val">▼{s2_azalma:.1f}%</div>
          <div class="lbl">Mevcut Duruma Göre</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card" style="--accent:#1B7837">
          <div class="lbl">S3 – Tam EV Emisyon Azalması</div>
          <div class="val">▼{s3_azalma:.1f}%</div>
          <div class="lbl">Mevcut Duruma Göre</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Kazanan
    st.markdown(f"""
    <div class="winner-box">
        <div class="wlbl">🏆 ÖNERİLEN OPTİMAL SENARYO</div>
        <div class="wval">{ETIKET[en_iyi]}</div>
        <div style="font-size:0.85rem; color:#8b949e; margin-top:6px;">
            Emisyon azaltımı ve toplam maliyet kriterleri %50-%50 ağırlıklandırılarak yapılan analitik hiyerarşi süreci (AHP) sonucunda en dengeli geçiş stratejisi seçilmiştir.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── SEKMELERs ──
    tab_emisyon, tab_maliyet, tab_basabas, tab_tablo = st.tabs([
        "♻️ EMİSYON ANALİZİ",
        "💹 MALİYET ANALİZİ",
        "📉 BAŞABAŞ ANALİZİ",
        "📊 DETAY TABLOLAR",
    ])

    # ──────────────────────────────────────────
    #  EMİSYON SEKMESİ
    # ──────────────────────────────────────────
    with tab_emisyon:
        st.subheader("IPCC Tier 2 – Senaryo Bazlı Yıllık Emisyon Karşılaştırması")
        senaryolar = ["MD", "S1", "S2", "S3"]
        renkler    = [RENK[k] for k in senaryolar]
        em_tum = [em_md, em_s1, em_s2, em_s3]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle("IPCC Tier 2 – Senaryo Bazlı Yıllık Emisyon Karşılaştırması", fontsize=12, fontweight="bold")
        for ax, key, birim, fmt, bolucu in [
            (axes[0,0], "CO2_kg",   "CO₂ (ton/yıl)",  ",.0f", 1000),
            (axes[0,1], "CH4_kg",   "CH₄ (kg/yıl)",   ",.3f", 1),
            (axes[1,0], "N2O_kg",   "N₂O (kg/yıl)",   ",.3f", 1),
            (axes[1,1], "CO2e_ton", "CO₂e (ton/yıl)", ",.1f", 1),
        ]:
            vals = [e[key] / bolucu for e in em_tum]
            bars = ax.bar(["MD","S1","S2","S3"], vals, color=renkler, width=0.5, edgecolor="white", linewidth=1.2)
            ax.set_title(key.split("_")[0] + " Emisyonu", fontweight="bold")
            ax.set_ylabel(birim)
            mx = max(vals) if max(vals) > 0 else 1
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + mx*0.01,
                        format(v, fmt), ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_ylim(0, mx * 1.18)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        st.markdown("#### Emisyon Karşılaştırma Tablosu")
        tablo_data = {
            "Gösterge": ["CO₂ (ton/yıl)", "CH₄ (kg/yıl)", "N₂O (kg/yıl)", "CO₂e (ton/yıl)"],
            "MD": [f"{em_md['CO2_kg']/1000:,.1f}", f"{em_md['CH4_kg']:,.3f}", f"{em_md['N2O_kg']:,.3f}", f"{em_md['CO2e_ton']:,.2f}"],
            "S1": [f"{em_s1['CO2_kg']/1000:,.1f}", f"{em_s1['CH4_kg']:,.3f}", f"{em_s1['N2O_kg']:,.3f}", f"{em_s1['CO2e_ton']:,.2f}"],
            "S1 Azalma": [f"▼%{(1-em_s1['CO2_kg']/em_md['CO2_kg'])*100:.1f}" if em_md['CO2_kg']>0 else "-",
                          f"▼%{(1-em_s1['CH4_kg']/em_md['CH4_kg'])*100:.1f}" if em_md['CH4_kg']>0 else "-",
                          f"▼%{(1-em_s1['N2O_kg']/em_md['N2O_kg'])*100:.1f}" if em_md['N2O_kg']>0 else "-",
                          f"▼%{(1-em_s1['CO2e_ton']/em_md['CO2e_ton'])*100:.1f}" if em_md['CO2e_ton']>0 else "-"],
            "S2": [f"{em_s2['CO2_kg']/1000:,.1f}", f"{em_s2['CH4_kg']:,.3f}", f"{em_s2['N2O_kg']:,.3f}", f"{em_s2['CO2e_ton']:,.2f}"],
            "S2 Azalma": [f"▼%{(1-em_s2['CO2_kg']/em_md['CO2_kg'])*100:.1f}" if em_md['CO2_kg']>0 else "-",
                          f"▼%{(1-em_s2['CH4_kg']/em_md['CH4_kg'])*100:.1f}" if em_md['CH4_kg']>0 else "-",
                          f"▼%{(1-em_s2['N2O_kg']/em_md['N2O_kg'])*100:.1f}" if em_md['N2O_kg']>0 else "-",
                          f"▼%{(1-em_s2['CO2e_ton']/em_md['CO2e_ton'])*100:.1f}" if em_md['CO2e_ton']>0 else "-"],
            "S3": [f"{em_s3['CO2_kg']/1000:,.1f}", f"{em_s3['CH4_kg']:,.3f}", f"{em_s3['N2O_kg']:,.3f}", f"{em_s3['CO2e_ton']:,.2f}"],
            "S3 Azalma": [f"▼%{(1-em_s3['CO2_kg']/em_md['CO2_kg'])*100:.1f}" if em_md['CO2_kg']>0 else "-",
                          f"▼%{(1-em_s3['CH4_kg']/em_md['CH4_kg'])*100:.1f}" if em_md['CH4_kg']>0 else "-",
                          f"▼%{(1-em_s3['N2O_kg']/em_md['N2O_kg'])*100:.1f}" if em_md['N2O_kg']>0 else "-",
                          f"▼%{(1-em_s3['CO2e_ton']/em_md['CO2e_ton'])*100:.1f}" if em_md['CO2e_ton']>0 else "-"],
        }
        st.dataframe(pd.DataFrame(tablo_data), use_container_width=True, hide_index=True)
        st.info("Azalma değerleri mevcut duruma (MD) göre kıyaslanmıştır.")

    # ──────────────────────────────────────────
    #  MALİYET SEKMESİ – Kümülatif Maliyet Tablosu + Grafik
    # ──────────────────────────────────────────
    with tab_maliyet:
        st.subheader(f"Kümülatif Maliyet Analizi – {ANALIZ_YILI} Yıl | {odeme_plani_adi}")

        # Yıllık kümülatif toplamlar tablosu
        yillar = list(range(1, ANALIZ_YILI + 1))

        def yillik_kumulatif(df):
            cum = []
            for y in yillar:
                cum.append(df[df["yil"] <= y]["toplam"].sum())
            return cum

        cum_md = yillik_kumulatif(df_md)
        cum_s1 = yillik_kumulatif(df_s1)
        cum_s2 = yillik_kumulatif(df_s2)
        cum_s3 = yillik_kumulatif(df_s3)

        tablo_kum = pd.DataFrame({
            "Yıl": yillar,
            "MD – Kümülatif Maliyet (TL)": [f"{v:,.0f}" for v in cum_md],
            "S1 – Kümülatif Maliyet (TL)": [f"{v:,.0f}" for v in cum_s1],
            "S2 – Kümülatif Maliyet (TL)": [f"{v:,.0f}" for v in cum_s2],
            "S3 – Kümülatif Maliyet (TL)": [f"{v:,.0f}" for v in cum_s3],
        })
        st.dataframe(tablo_kum, use_container_width=True, hide_index=True)

        # Kümülatif maliyet çizgi grafiği
        fig_km, ax_km = plt.subplots(figsize=(13, 5))
        ax_km.plot(yillar, [v/1e6 for v in cum_md], color=RENK["MD"], lw=2.5, marker="o", markersize=4, label="MD – Mevcut Durum")
        ax_km.plot(yillar, [v/1e6 for v in cum_s1], color=RENK["S1"], lw=2.5, marker="s", markersize=4, label="S1 – 1/3 EV")
        ax_km.plot(yillar, [v/1e6 for v in cum_s2], color=RENK["S2"], lw=2.5, marker="^", markersize=4, label="S2 – 2/3 EV")
        ax_km.plot(yillar, [v/1e6 for v in cum_s3], color=RENK["S3"], lw=2.5, marker="D", markersize=4, label="S3 – Tam EV")
        ax_km.set_title(f"Senaryo Bazlı Kümülatif Toplam Maliyet – {ANALIZ_YILI} Yıl", fontweight="bold")
        ax_km.set_xlabel("Yıl"); ax_km.set_ylabel("Kümülatif Maliyet (Milyon TL)")
        ax_km.xaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_km.grid(True, linestyle=":", alpha=0.5)
        ax_km.legend()
        plt.tight_layout()
        st.pyplot(fig_km); plt.close(fig_km)

        # Aylık maliyet bileşenleri grafikleri
        st.markdown("---")
        st.subheader("Senaryo Bazlı Aylık Maliyet Bileşenleri")
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
                    plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
                with col_t:
                    yillik_df = df_.groupby("yil")[["yakıt","bakım","taksit","toplam"]].mean().reset_index()
                    yillik_df.columns = ["Yıl","Yakıt (TL)","Bakım (TL)","Taksit (TL)","Toplam (TL)"]
                    for c in ["Yakıt (TL)","Bakım (TL)","Taksit (TL)","Toplam (TL)"]:
                        yillik_df[c] = yillik_df[c].map(lambda x: f"{x:,.0f}")
                    yillik_df["Yıl"] = yillik_df["Yıl"].astype(int)
                    st.dataframe(yillik_df, use_container_width=True, hide_index=True,
                                 height=min(40 + ANALIZ_YILI * 35, 500))

    # ──────────────────────────────────────────
    #  BAŞABAŞ ANALİZİ SEKMESİ – S1, S2, S3 AYRI GRAFİK
    # ──────────────────────────────────────────
    with tab_basabas:
        st.subheader(f"📉 Başabaş Analizi – Senaryo Bazlı Kümülatif Maliyet Karşılaştırması")
        st.caption(f"Ödeme Planı: {odeme_plani_adi} | Analiz Süresi: {ANALIZ_YILI} Yıl")
        st.markdown("""
        <div class="info-box">
        <b>Grafik Açıklaması:</b>
        <b>MD (gri) eğrisi</b> 0'dan başlar; her ay birikimli dizel işletme maliyeti (yakıt + bakım) eklenerek yükselir.
        <b>Senaryo (renkli) eğrisi</b> araç yatırım maliyetiyle başlar; üzerine EV işletme maliyeti (yakıt + bakım) aylık eklenir.
        EV işletme maliyeti daha düşük olduğundan senaryo eğrisi daha yavaş yükselir ve bir noktada MD eğrisini keser.
        <b>Kesişme noktası = başabaş noktası</b>: bu andan sonra EV geçişi toplam maliyette MD'den daha avantajlı olur.
        </div>
        """, unsafe_allow_html=True)

        senaryo_liste = [
            ("S1", df_s1, yat1, "1/3 EV Geçişi"),
            ("S2", df_s2, yat2, "2/3 EV Geçişi"),
            ("S3", df_s3, yat3, "Tam EV Geçişi"),
        ]

        for kod, df_sc, yatirim, aciklama in senaryo_liste:
            st.markdown(f"### {ETIKET[kod]}")

            toplam_ay = ANALIZ_YILI * 12

            # ── Başabaş eğrilerini ay ay inşa et ──
            # MD eğrisi  : ay 0'da 0 TL, her ay o ayın dizel işletme maliyeti (yakıt+bakım) eklenir
            # Sx eğrisi  : ay 0'da yatırım tutarı, her ay o ayın EV+karma işletme maliyeti (yakıt+bakım) eklenir
            # İki eğri kesiştiğinde = başabaş noktası
            #
            # Neden başabaş noktaları farklı çıkmalı?
            #   S1 az yatırım (düşük başlangıç) ama az tasarruf (1/3 EV)  → orta vadede kesişir
            #   S2 orta yatırım, orta tasarruf (2/3 EV)
            #   S3 çok yatırım (yüksek başlangıç) ama çok tasarruf (tam EV) → farklı noktada kesişir

            aylar_grafik = list(range(0, toplam_ay + 1))   # 0..120 (ay 0 = başlangıç)

            # MD: ay 0'da 0, sonraki aylarda birikimli işletme maliyeti
            md_kumulatif = [0.0]
            for i in range(toplam_ay):
                md_kumulatif.append(md_kumulatif[-1] + df_md["yakıt"].iloc[i] + df_md["bakım"].iloc[i])

            # Sx: ay 0'da yatırım tutarı, sonraki aylarda birikimli EV+karma işletme maliyeti eklenir
            sc_kumulatif = [float(yatirim)]
            for i in range(toplam_ay):
                sc_kumulatif.append(sc_kumulatif[-1] + df_sc["yakıt"].iloc[i] + df_sc["bakım"].iloc[i])

            md_arr = np.array(md_kumulatif)
            sc_arr = np.array(sc_kumulatif)

            # Başabaş: MD eğrisi Sx eğrisini geçtiği (MD >= Sx) ilk ay indeksi
            idx_be = np.where(md_arr >= sc_arr)[0]

            fig_bb, ax_bb = plt.subplots(figsize=(13, 5))

            ax_bb.plot(aylar_grafik, md_arr / 1e6,
                       color=RENK["MD"], lw=2.5,
                       label="MD – Kümülatif Dizel İşletme Maliyeti (0'dan başlar)")
            ax_bb.plot(aylar_grafik, sc_arr / 1e6,
                       color=RENK[kod], lw=2.5, linestyle="--",
                       label=f"{ETIKET[kod]} – Araç Yatırımı + Kümülatif EV İşletme Maliyeti")

            # Ay 0 yatırım noktası
            ax_bb.scatter([0], [yatirim / 1e6], color=RENK[kod], s=100, zorder=6,
                          label=f"Başlangıç Yatırımı: {yatirim/1e6:,.1f} M TL")

            net_son = (md_arr[-1] - sc_arr[-1]) / 1e6

            if len(idx_be) > 0:
                be_idx  = int(idx_be[0])
                be_ay   = be_idx          # ay 0 bazlı
                be_yil  = be_ay / 12
                be_val  = md_arr[be_idx] / 1e6   # kesişme noktasındaki TL değeri

                ax_bb.scatter([be_ay], [be_val], color="red", s=180, zorder=7, marker="*")

                # Etiket: grafiğin sağına taşma yapmadan konumlandır
                x_text = min(be_ay + max(3, ANALIZ_YILI * 0.6), toplam_ay - 5)
                y_text = be_val * 1.08
                ax_bb.annotate(
                    f"BAŞABAŞ\n{be_ay}. Ay ({be_yil:.1f} yıl)\n{be_val:,.1f} M TL",
                    xy=(be_ay, be_val),
                    xytext=(x_text, y_text),
                    color="red", fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.4, alpha=0.8),
                    bbox=dict(facecolor="white", alpha=0.9, edgecolor="red", boxstyle="round,pad=0.3")
                )
                # Başabaş sonrası tasarruf bölgesi (MD üstte, Sx altta)
                ax_bb.fill_between(
                    aylar_grafik[be_idx:],
                    md_arr[be_idx:] / 1e6,
                    sc_arr[be_idx:] / 1e6,
                    alpha=0.18, color="#1B7837", label=f"Net Tasarruf Bölgesi ({net_son:+.1f} M TL)"
                )
                ax_bb.set_title(
                    f"{ETIKET[kod]} – Başabaş Analizi  |  "
                    f"Başabaş: {be_ay}. Ay ({be_yil:.1f} Yıl)  |  "
                    f"{ANALIZ_YILI} Yıl Sonu Net: {net_son:+,.1f} M TL",
                    fontweight="bold", fontsize=11
                )
            else:
                ax_bb.set_title(
                    f"{ETIKET[kod]} – Başabaş Analizi\n"
                    f"⚠️ {ANALIZ_YILI} yıl içinde başabaş noktasına ulaşılamadı  |  "
                    f"{ANALIZ_YILI} Yıl Sonu Net: {net_son:+,.1f} M TL",
                    fontweight="bold", fontsize=11
                )

            # Yıl çizgileri
            y_min = min(0, sc_arr.min() / 1e6) * 0.98
            for y in range(1, ANALIZ_YILI + 1):
                ax_bb.axvline(y * 12, color="gray", lw=0.4, alpha=0.25)
                ax_bb.text(y * 12 + 0.3, y_min, f"Y{y}", fontsize=6, color="#888", va="bottom")

            ax_bb.set_xlim(0, toplam_ay + 2)
            ax_bb.set_xlabel("Zaman (Ay)", fontweight="bold")
            ax_bb.set_ylabel("Kümülatif Ödenen Tutar (Milyon TL)", fontweight="bold")
            ax_bb.xaxis.set_major_locator(mticker.MultipleLocator(12))
            ax_bb.grid(True, linestyle=":", alpha=0.4)
            ax_bb.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig_bb); plt.close(fig_bb)

            # Kısa özet
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"""<div class="metric-card" style="--accent:{RENK[kod]}">
                  <div class="lbl">Toplam Yatırım</div>
                  <div class="val">{yatirim/1e6:,.1f} M TL</div>
                  <div class="lbl">Araç Alım Maliyeti</div></div>""", unsafe_allow_html=True)
            with col_b:
                if len(idx_be) > 0:
                    st.markdown(f"""<div class="metric-card" style="--accent:#1B7837">
                      <div class="lbl">Başabaş Noktası</div>
                      <div class="val">{be_ay}. Ay</div>
                      <div class="lbl">{be_yil:.1f} yıl sonra</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="metric-card" style="--accent:#cc3300">
                      <div class="lbl">Başabaş Noktası</div>
                      <div class="val">—</div>
                      <div class="lbl">{ANALIZ_YILI} yılda ulaşılamadı</div></div>""", unsafe_allow_html=True)
            with col_c:
                st.markdown(f"""<div class="metric-card" style="--accent:{RENK[kod]}">
                  <div class="lbl">{ANALIZ_YILI} Yıl Sonu Net Fark</div>
                  <div class="val">{net_son:+,.1f} M TL</div>
                  <div class="lbl">+ Tasarruf / - Zarar</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

    # ──────────────────────────────────────────
    #  DETAY TABLOLAR SEKMESİ
    # ──────────────────────────────────────────
    with tab_tablo:
        st.subheader("Senaryolara Ait Aylık Ham Veri Çıktıları")
        secilen_kod = st.selectbox("Senaryo Seçin:", ["MD", "S1", "S2", "S3"])
        orijinal_df = {"MD": df_md, "S1": df_s1, "S2": df_s2, "S3": df_s3}[secilen_kod]
        gosterim_df = orijinal_df.copy()
        gosterim_df.columns = ["Ay", "Yıl", "Yakıt (TL)", "Bakım (TL)", "Taksit (TL)", "İşletme (TL)", "Toplam (TL)"]
        st.dataframe(gosterim_df.style.format({
            "Yakıt (TL)": "{:,.2f}", "Bakım (TL)": "{:,.2f}",
            "Taksit (TL)": "{:,.2f}", "İşletme (TL)": "{:,.2f}", "Toplam (TL)": "{:,.2f}"
        }), use_container_width=True, hide_index=True)
