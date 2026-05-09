"""
ELEKTRİKLİ ARACA GEÇİŞ SÜRECİ İÇİN KARAR DESTEK SİSTEMİ
IPCC Tier 2 Metodolojisi & XGBoost Destekli Senaryo Analizi
Karabük Üniversitesi – Endüstri Mühendisliği Lisans Bitirme Tezi
Özge ÖZBAY & Sümeyye TEKİN

Streamlit Uygulaması
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  SAYFA YAPILANDIRMASI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EV Geçiş Karar Destek Sistemi",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  ÖZEL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 2px solid #1e9e6b;
}
section[data-testid="stSidebar"] * {
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stRadio label {
    color: #8b949e !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #1e9e6b !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em;
}

/* Ana arka plan */
.stApp {
    background: #f7f8fc;
}

/* Başlık bloğu */
.hero-block {
    background: linear-gradient(135deg, #0d1117 60%, #1a2a1e 100%);
    color: #e6edf3;
    padding: 2.2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border-left: 5px solid #1e9e6b;
    position: relative;
    overflow: hidden;
}
.hero-block::after {
    content: '⚡';
    position: absolute;
    right: 2rem;
    top: 1.5rem;
    font-size: 3.5rem;
    opacity: 0.12;
}
.hero-block h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    margin: 0 0 0.3rem 0;
    color: #1e9e6b;
    letter-spacing: 0.04em;
}
.hero-block p {
    font-size: 0.88rem;
    color: #8b949e;
    margin: 0;
}
.hero-block .badge {
    display: inline-block;
    background: #1e9e6b22;
    border: 1px solid #1e9e6b55;
    color: #1e9e6b;
    border-radius: 4px;
    font-size: 0.72rem;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.6rem;
    margin-right: 4px;
}

/* Metrik kartları */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 160px;
    background: #fff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #e1e4e8;
    border-top: 3px solid var(--accent, #1e9e6b);
    box-shadow: 0 1px 4px #0001;
}
.metric-card .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #0d1117;
    line-height: 1.1;
}
.metric-card .lbl {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6e7781;
    margin-top: 4px;
}

/* Senaryo sekmeleri */
.tab-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
}

/* Sonuç kartları */
.result-card {
    background: #fff;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #e1e4e8;
    margin-bottom: 0.8rem;
}
.result-card h4 {
    margin: 0 0 0.6rem 0;
    font-size: 0.85rem;
    color: #6e7781;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Uyarı / öneri kutusu */
.winner-box {
    background: linear-gradient(135deg, #1a2a1e, #0d1117);
    border: 1px solid #1e9e6b;
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    color: #e6edf3;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 1rem;
}
.winner-box .wlbl {
    color: #1e9e6b;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.winner-box .wval {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 4px;
}

/* Tablo stili */
.stDataFrame {
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Buton */
.stButton > button {
    background: #1e9e6b !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #17845a !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px #1e9e6b44 !important;
}

/* Divider */
hr {
    border-color: #e1e4e8 !important;
    margin: 1.5rem 0 !important;
}

/* Info box */
.info-box {
    background: #ddf4e844;
    border-left: 4px solid #1e9e6b;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #1a3a2a;
    margin: 0.8rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SABİT DEĞERLER (IPCC Tier 2)
# ─────────────────────────────────────────────
EF_CO2_DIZEL  = 2.690   # kg/L
EF_CO2_BENZIN = 2.350   # kg/L
EF_CH4_OTOBÜS_DIZEL  = 3.9   # mg/km
EF_N2O_OTOBÜS_DIZEL  = 3.9   # mg/km
EF_CH4_MINİBÜS_DIZEL = 3.9   # mg/km
EF_N2O_MINİBÜS_DIZEL = 3.9   # mg/km
EF_GRID      = 0.43    # kg CO2/kWh (IEA 2023, Türkiye)
ETA_SARJ     = 0.90
E_OTOBÜS_EV  = 0.18    # kWh/km
E_MINİBÜS_EV = 0.12    # kWh/km
TUK_OTOBÜS_DIZEL  = 0.33  # L/km
TUK_MINİBÜS_DIZEL = 0.12  # L/km
GWP_CH4 = 28
GWP_N2O = 265

RENK = {"S1": "#2166AC", "S2": "#F4A100", "S3": "#1B7837"}
ETIKET = {
    "S1": "Senaryo 1 – Mevcut Durum (Tam Dizel)",
    "S2": "Senaryo 2 – Kısmi EV Geçişi",
    "S3": "Senaryo 3 – Tam EV Geçişi",
}

# ─────────────────────────────────────────────
#  BAŞLIK
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-block">
  <h1>🚌 ELEKTRİKLİ ARACA GEÇİŞ KARAR DESTEK SİSTEMİ</h1>
  <p>Karabük UlaşımAŞ Toplu Taşıma Filosu · Senaryo Analizi </p>
  <span class="badge">IPCC Tier 2</span>
  <span class="badge">Karabük Üniversitesi</span>
  <span class="badge">Endüstri Mühendisliği</span>
  <span class="badge">Özge ÖZBAY & Sümeyye TEKİN</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR – GİRİŞ PARAMETRELERİ
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📋 FİLO BİLGİLERİ")
    st.markdown("### MEVCUT FİLO")
    n_otobüs_mevcut  = st.number_input("Dizel Otobüs Sayısı (adet)", min_value=1, value=20, step=1)
    n_minibüs_mevcut = st.number_input("Dizel Minibüs Sayısı (adet)", min_value=0, value=10, step=1)

    st.markdown("### ELEKTRİKLİ GEÇİŞ PLANI")
    n_otobüs_ev  = st.number_input("EV'e Dönüşecek Otobüs Sayısı", min_value=0,
                                    max_value=int(n_otobüs_mevcut), value=min(10, int(n_otobüs_mevcut)), step=1)
    n_minibüs_ev = st.number_input("EV'e Dönüşecek Minibüs Sayısı", min_value=0,
                                    max_value=int(n_minibüs_mevcut), value=min(5, int(n_minibüs_mevcut)), step=1)

    st.markdown("### ARAÇ FİYATLARI (TL)")
    fiyat_otobüs_ev  = st.number_input("Elektrikli Otobüs Birim Fiyatı (TL)", min_value=1.0,
                                        value=8_000_000.0, step=100_000.0, format="%.0f")
    fiyat_minibüs_ev = st.number_input("Elektrikli Minibüs Birim Fiyatı (TL)", min_value=0.0,
                                        value=3_500_000.0, step=100_000.0, format="%.0f")

    st.markdown("### BAKIM MALİYETLERİ (TL/araç/yıl)")
    bakim_otobüs_dizel  = st.number_input("Dizel Otobüs Bakım", min_value=0.0, value=150_000.0, step=10_000.0, format="%.0f")
    bakim_minibüs_dizel = st.number_input("Dizel Minibüs Bakım", min_value=0.0, value=80_000.0,  step=10_000.0, format="%.0f")
    bakim_otobüs_ev     = st.number_input("EV Otobüs Bakım",     min_value=0.0, value=60_000.0,  step=10_000.0, format="%.0f")
    bakim_minibüs_ev    = st.number_input("EV Minibüs Bakım",    min_value=0.0, value=35_000.0,  step=10_000.0, format="%.0f")

    st.markdown("### YAKIT / ENERJİ FİYATLARI")
    dizel_fiyat    = st.number_input("Dizel Fiyatı (TL/L)",    min_value=0.0, value=45.0, step=1.0)
    elektrik_fiyat = st.number_input("Elektrik Fiyatı (TL/kWh)", min_value=0.0, value=4.5, step=0.1)

    st.markdown("### YILLIK KİLOMETRELER")
    km_otobüs_yillik  = st.number_input("Otobüs Filosu Toplam Yıllık km", min_value=1.0,
                                         value=1_500_000.0, step=10_000.0, format="%.0f")
    km_minibüs_yillik = st.number_input("Minibüs Filosu Toplam Yıllık km", min_value=0.0,
                                         value=600_000.0,   step=10_000.0, format="%.0f")

    st.markdown("### ENFLASYON & ÖDEME")
    tufe_yuzde  = st.number_input("Yıllık TÜFE Oranı (%)", min_value=0.0, value=30.0, step=1.0)
    tufe_orani  = tufe_yuzde / 100.0

    ANALIZ_YILI = st.number_input("Ödeme Planı Süresi (yıl)", min_value=1, value=15, step=1)

    odeme_plani = st.radio(
        "Ödeme Planı",
        options=[1, 2],
        format_func=lambda x: "Sabit Yıllık Ödeme" if x == 1 else f"TÜİK Zam Bazlı (%{tufe_yuzde:.0f}/yıl)",
        index=0,
    )
    odeme_plani_adi = "Sabit Ödeme Planı" if odeme_plani == 1 else f"TÜİK Zam Oranı Bazlı Plan (%{tufe_yuzde:.0f}/yıl)"

   

    hesapla_btn = st.button("🔍 ANALİZİ ÇALIŞTIR", use_container_width=True)

# ─────────────────────────────────────────────
#  HESAPLAMALAR
# ─────────────────────────────────────────────

def run_analysis():
    # Senaryo tanımları
    s1 = dict(otobüs_dizel=n_otobüs_mevcut, otobüs_ev=0,
               minibüs_dizel=n_minibüs_mevcut, minibüs_ev=0)
    s2 = dict(otobüs_dizel=n_otobüs_mevcut - n_otobüs_ev, otobüs_ev=n_otobüs_ev,
               minibüs_dizel=n_minibüs_mevcut - n_minibüs_ev, minibüs_ev=n_minibüs_ev)
    s3 = dict(otobüs_dizel=0, otobüs_ev=n_otobüs_mevcut,
               minibüs_dizel=0, minibüs_ev=n_minibüs_mevcut)

    def emisyon_hesapla(senaryo, km_oto, km_mini):
        r = {}
        oto_km_d  = (senaryo["otobüs_dizel"] / max(n_otobüs_mevcut, 1)) * km_oto
        co2_od    = oto_km_d * TUK_OTOBÜS_DIZEL * EF_CO2_DIZEL
        ch4_od    = oto_km_d * EF_CH4_OTOBÜS_DIZEL / 1e6
        n2o_od    = oto_km_d * EF_N2O_OTOBÜS_DIZEL / 1e6
        oto_km_ev = (senaryo["otobüs_ev"] / max(n_otobüs_mevcut, 1)) * km_oto
        co2_oev   = oto_km_ev * (E_OTOBÜS_EV / ETA_SARJ) * EF_GRID

        mini_km_d  = (senaryo["minibüs_dizel"] / max(n_minibüs_mevcut, 1)) * km_mini if n_minibüs_mevcut > 0 else 0
        co2_md     = mini_km_d * TUK_MINİBÜS_DIZEL * EF_CO2_DIZEL
        ch4_md     = mini_km_d * EF_CH4_MINİBÜS_DIZEL / 1e6
        n2o_md     = mini_km_d * EF_N2O_MINİBÜS_DIZEL / 1e6
        mini_km_ev = (senaryo["minibüs_ev"] / max(n_minibüs_mevcut, 1)) * km_mini if n_minibüs_mevcut > 0 else 0
        co2_mev    = mini_km_ev * (E_MINİBÜS_EV / ETA_SARJ) * EF_GRID

        r["CO2_kg"]   = co2_od + co2_oev + co2_md + co2_mev
        r["CH4_kg"]   = ch4_od + ch4_md
        r["N2O_kg"]   = n2o_od + n2o_md
        r["CO2e_ton"] = (r["CO2_kg"] + r["CH4_kg"] * GWP_CH4 + r["N2O_kg"] * GWP_N2O) / 1000
        return r

    em_s1 = emisyon_hesapla(s1, km_otobüs_yillik, km_minibüs_yillik)
    em_s2 = emisyon_hesapla(s2, km_otobüs_yillik, km_minibüs_yillik)
    em_s3 = emisyon_hesapla(s3, km_otobüs_yillik, km_minibüs_yillik)

    def maliyet_serileri(senaryo, n_arac_ev_oto, n_arac_ev_mini,
                          fiyat_oto_ev, fiyat_mini_ev, tufe, yil, odeme_plan):
        ayl_yakıt_d = (
            senaryo["otobüs_dizel"]  * (km_otobüs_yillik  / n_otobüs_mevcut  if n_otobüs_mevcut  else 0) * TUK_OTOBÜS_DIZEL  * dizel_fiyat +
            senaryo["minibüs_dizel"] * (km_minibüs_yillik / n_minibüs_mevcut if n_minibüs_mevcut else 0) * TUK_MINİBÜS_DIZEL * dizel_fiyat
        ) / 12
        ayl_yakıt_ev = (
            senaryo["otobüs_ev"]  * (km_otobüs_yillik  / n_otobüs_mevcut  if n_otobüs_mevcut  else 0) * (E_OTOBÜS_EV  / ETA_SARJ) * elektrik_fiyat +
            senaryo["minibüs_ev"] * (km_minibüs_yillik / n_minibüs_mevcut if n_minibüs_mevcut else 0) * (E_MINİBÜS_EV / ETA_SARJ) * elektrik_fiyat
        ) / 12
        ayl_bakım = (
            senaryo["otobüs_dizel"]  * bakim_otobüs_dizel  +
            senaryo["minibüs_dizel"] * bakim_minibüs_dizel +
            senaryo["otobüs_ev"]     * bakim_otobüs_ev     +
            senaryo["minibüs_ev"]    * bakim_minibüs_ev
        ) / 12
        yatirim = n_arac_ev_oto * fiyat_oto_ev + n_arac_ev_mini * fiyat_mini_ev
        if odeme_plan == 1:
            taksit_sabit = yatirim / (yil * 12) if yatirim > 0 and yil > 0 else 0
        else:
            if tufe > 0:
                carpan_t = sum((1 + tufe) ** t for t in range(yil))
                taksit_y1y = yatirim / carpan_t if carpan_t > 0 else 0
            else:
                taksit_y1y = yatirim / yil if yil > 0 else 0

        kayitlar = []
        for ay in range(1, yil * 12 + 1):
            yn = (ay - 1) // 12
            yc = (1 + tufe) ** yn
            yakıt = (ayl_yakıt_d + ayl_yakıt_ev) * yc
            bakım = ayl_bakım * yc
            taksit = (taksit_sabit if odeme_plan == 1
                      else ((taksit_y1y / 12) * yc if yatirim > 0 else 0))
            kayitlar.append({"ay": ay, "yil": yn + 1, "yakıt": yakıt,
                              "bakım": bakım, "taksit": taksit, "toplam": yakıt + bakım + taksit})
        return pd.DataFrame(kayitlar)

    df_s1 = maliyet_serileri(s1, 0, 0, 0, 0, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s2 = maliyet_serileri(s2, n_otobüs_ev, n_minibüs_ev, fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s3 = maliyet_serileri(s3, n_otobüs_mevcut, n_minibüs_mevcut, fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)

   
# ─────────────────────────────────────────────
#  BAŞLANGIÇ YA DA HESAP SONRASI GÖSTERİM
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None

if hesapla_btn:
    with st.spinner("Analiz çalışıyor…"):
        st.session_state["results"] = run_analysis()

res = st.session_state["results"]

# ─────────────────────────────────────────────
#  SONUÇLAR
# ─────────────────────────────────────────────
if res is None:
    st.markdown("""
    <div class="info-box">
    ℹ️ Sol panelden filo bilgilerini ve parametrelerinizi girdikten sonra <b>ANALİZİ ÇALIŞTIR</b> butonuna tıklayın.
    </div>
    """, unsafe_allow_html=True)

    # Parametreler bilgi kartı
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card" style="--accent:#2166AC">
          <div class="lbl">IPCC Metodolojisi</div>
          <div class="val" style="font-size:1.1rem">Tier 2</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="--accent:#1e9e6b">
          <div class="lbl">Şebeke EF (Türkiye)</div>
          <div class="val">0.43 <span style="font-size:0.8rem">kg CO₂/kWh</span></div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="--accent:#F4A100">
          <div class="lbl">Şarj Verimi</div>
          <div class="val">%90</div>
        </div>""", unsafe_allow_html=True)


    em_listesi = [em_s1, em_s2, em_s3]
    df_listesi = [df_s1, df_s2, df_s3]

    # ── Özet metrik kartları ──
    col1, col2, col3, col4 = st.columns(4)
    s2_azalma = (1 - em_s2["CO2e_ton"] / em_s1["CO2e_ton"]) * 100 if em_s1["CO2e_ton"] > 0 else 0
    s3_azalma = (1 - em_s3["CO2e_ton"] / em_s1["CO2e_ton"]) * 100 if em_s1["CO2e_ton"] > 0 else 0
    with col1:
        st.markdown(f"""<div class="metric-card" style="--accent:#2166AC">
          <div class="lbl">S1 Yıllık Emisyon</div>
          <div class="val">{em_s1['CO2e_ton']:,.0f}</div>
          <div class="lbl">ton CO₂e/yıl</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="--accent:#F4A100">
          <div class="lbl">S2 Emisyon Azalması</div>
          <div class="val">▼{s2_azalma:.1f}%</div>
          <div class="lbl">S1'e kıyasla</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="--accent:#1B7837">
          <div class="lbl">S3 Emisyon Azalması</div>
          <div class="val">▼{s3_azalma:.1f}%</div>
          <div class="lbl">S1'e kıyasla</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card" style="--accent:#1e9e6b">
          <div class="lbl">AHP Önerisi</div>
          <div class="val" style="font-size:1rem">{en_iyi}</div>
          <div class="lbl">{ETIKET[en_iyi]}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Ana sekmeler ──
    tab_emisyon, tab_maliyet, tab_kumulatif, tab_tablo = st.tabs([
        "🌿 EMİSYON ANALİZİ",
        "💰 MALİYET ANALİZİ",
        "📈 KÜMÜLATİF MALİYET",
        "📊 DETAY TABLOLAR",
    ])

    # ──────────────────────────────────────────
    #  EMİSYON SEKMESİ
    # ──────────────────────────────────────────
    with tab_emisyon:
        st.subheader("IPCC Tier 2 – Senaryo Bazlı Yıllık Emisyon Karşılaştırması")

        plt.rcParams.update({
            "font.family": "DejaVu Sans", "axes.titlesize": 12,
            "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
            "legend.fontsize": 8, "axes.grid": True, "grid.alpha": 0.3,
            "axes.spines.top": False, "axes.spines.right": False,
        })

        senaryolar = ["S1", "S2", "S3"]
        etiketler  = [ETIKET[k] for k in senaryolar]
        renkler    = [RENK[k] for k in senaryolar]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            "IPCC Tier 2 – Senaryo Bazlı Yıllık Emisyon Karşılaştırması\nKarabük UlaşımAŞ Toplu Taşıma Filosu",
            fontsize=12, fontweight="bold"
        )
        for ax, key, birim, fmt, bolucu in [
            (axes[0,0], "CO2_kg",   "CO₂ (ton/yıl)",  ",.0f", 1000),
            (axes[0,1], "CH4_kg",   "CH₄ (kg/yıl)",   ",.3f", 1),
            (axes[1,0], "N2O_kg",   "N₂O (kg/yıl)",   ",.3f", 1),
            (axes[1,1], "CO2e_ton", "CO₂e (ton/yıl)", ",.1f", 1),
        ]:
            vals = [e[key] / bolucu for e in em_listesi]
            bars = ax.bar(["S1","S2","S3"], vals, color=renkler, width=0.5, edgecolor="white", linewidth=1.2)
            ax.set_title(key.replace("_kg","").replace("_ton","").upper() + " Emisyonu", fontweight="bold")
            ax.set_ylabel(birim)
            mx = max(vals) if max(vals) > 0 else 1
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + mx*0.01,
                        format(v, fmt), ha="center", va="bottom", fontsize=8, fontweight="bold")
            ax.set_ylim(0, mx * 1.18)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Emisyon özet tablosu
        st.markdown("#### Emisyon Karşılaştırma Tablosu")
        tablo_data = {
            "Gösterge": ["CO₂ (ton/yıl)", "CH₄ (kg/yıl)", "N₂O (kg/yıl)", "CO₂e (ton/yıl)"],
            "S1 – Mevcut": [
                f"{em_s1['CO2_kg']/1000:,.1f}", f"{em_s1['CH4_kg']:,.3f}",
                f"{em_s1['N2O_kg']:,.3f}", f"{em_s1['CO2e_ton']:,.2f}"],
            "S2 – Kısmi EV": [
                f"{em_s2['CO2_kg']/1000:,.1f}", f"{em_s2['CH4_kg']:,.3f}",
                f"{em_s2['N2O_kg']:,.3f}", f"{em_s2['CO2e_ton']:,.2f}"],
            "S2 Azalma": [
                f"▼%{(1-em_s2['CO2_kg']/em_s1['CO2_kg'])*100:.1f}" if em_s1['CO2_kg']>0 else "-",
                f"▼%{(1-em_s2['CH4_kg']/em_s1['CH4_kg'])*100:.1f}" if em_s1['CH4_kg']>0 else "-",
                f"▼%{(1-em_s2['N2O_kg']/em_s1['N2O_kg'])*100:.1f}" if em_s1['N2O_kg']>0 else "-",
                f"▼%{(1-em_s2['CO2e_ton']/em_s1['CO2e_ton'])*100:.1f}" if em_s1['CO2e_ton']>0 else "-"],
            "S3 – Tam EV": [
                f"{em_s3['CO2_kg']/1000:,.1f}", f"{em_s3['CH4_kg']:,.3f}",
                f"{em_s3['N2O_kg']:,.3f}", f"{em_s3['CO2e_ton']:,.2f}"],
            "S3 Azalma": [
                f"▼%{(1-em_s3['CO2_kg']/em_s1['CO2_kg'])*100:.1f}" if em_s1['CO2_kg']>0 else "-",
                f"▼%{(1-em_s3['CH4_kg']/em_s1['CH4_kg'])*100:.1f}" if em_s1['CH4_kg']>0 else "-",
                f"▼%{(1-em_s3['N2O_kg']/em_s1['N2O_kg'])*100:.1f}" if em_s1['N2O_kg']>0 else "-",
                f"▼%{(1-em_s3['CO2e_ton']/em_s1['CO2e_ton'])*100:.1f}" if em_s1['CO2e_ton']>0 else "-"],
        }
        st.dataframe(pd.DataFrame(tablo_data), use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────
    #  MALİYET SEKMESİ
    # ──────────────────────────────────────────
    with tab_maliyet:
        st.subheader(f"Senaryo Bazlı Aylık Maliyet Analizi – {ANALIZ_YILI} Yıl")
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

    # ──────────────────────────────────────────
    #  KÜMÜLATİF MALİYET SEKMESİ
    # ──────────────────────────────────────────
    with tab_kumulatif:
        st.subheader(f"Senaryo Bazlı Kümülatif Maliyet Karşılaştırması – {ANALIZ_YILI} Yıl")

        fig3, ax3 = plt.subplots(figsize=(12, 5))
        for df_, kod in [(df_s1,"S1"), (df_s2,"S2"), (df_s3,"S3")]:
            cumul = df_["toplam"].cumsum() / 1e9
            ax3.plot(df_["ay"], cumul, color=RENK[kod], linewidth=2.5, label=ETIKET[kod])
        ax3.set_title(f"Kümülatif Toplam Maliyet ({ANALIZ_YILI} Yıl) | {odeme_plani_adi}", fontweight="bold")
        ax3.set_xlabel("Ay"); ax3.set_ylabel("Kümülatif Toplam Maliyet (Milyar TL)")
        ax3.legend(loc="upper left", fontsize=8)
        ax3.xaxis.set_major_locator(mticker.MultipleLocator(12))
        for y in range(1, ANALIZ_YILI+1):
            ax3.axvline(y*12, color="gray", lw=0.4, alpha=0.3, linestyle="--")
        plt.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

        # Özet toplam maliyet
        col_a, col_b, col_c = st.columns(3)
        for col, kod, df_ in [(col_a,"S1",df_s1),(col_b,"S2",df_s2),(col_c,"S3",df_s3)]:
            toplam_g = df_["toplam"].sum() / 1e9
            with col:
                st.markdown(f"""<div class="metric-card" style="--accent:{RENK[kod]}">
                  <div class="lbl">{ETIKET[kod]}</div>
                  <div class="val">{toplam_g:.2f} <span style="font-size:0.75rem">Milyar TL</span></div>
                  <div class="lbl">{ANALIZ_YILI} Yıl Toplam</div></div>""", unsafe_allow_html=True)

        # Yıllık toplam çubuk grafikleri
        st.markdown("#### Yıllık Toplam Maliyet Karşılaştırması")
        fig4, axes4 = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        for ax_, kod, df_ in zip(axes4, ["S1","S2","S3"], [df_s1,df_s2,df_s3]):
            yillik = df_.groupby("yil")["toplam"].sum().reset_index()
            bars = ax_.bar(yillik["yil"], yillik["toplam"]/1e6, color=RENK[kod], edgecolor="white", linewidth=1)
            ax_.set_title(ETIKET[kod], fontweight="bold", fontsize=8)
            ax_.set_xlabel("Yıl"); ax_.set_ylabel("Milyon TL")
            ax_.set_xticks(yillik["yil"])
            ax_.tick_params(axis='x', labelsize=7)
            mx = yillik["toplam"].max()/1e6
            for bar, v in zip(bars, yillik["toplam"]/1e6):
                ax_.text(bar.get_x()+bar.get_width()/2, v+mx*0.01, f"{v:.0f}M",
                          ha="center", va="bottom", fontsize=6)
        plt.tight_layout()
        st.pyplot(fig4); plt.close(fig4)

    
    # ──────────────────────────────────────────
    #  DETAY TABLOLAR SEKMESİ
    # ──────────────────────────────────────────
    with tab_tablo:
        st.subheader("Özet Rapor & Detay Tablolar")

        # Özet parametreler
        st.markdown("#### 🔧 Kullanılan Parametreler")
        params_df = pd.DataFrame({
            "Parametre": [
                "Ödeme Süresi", "Ödeme Planı", "Yıllık TÜFE", "Dizel Fiyatı",
                "Elektrik Fiyatı", "Şarj Verimi (η)", "Şebeke EF (Türkiye)",
                "AHP: Emisyon Ağırlığı", "AHP: Maliyet Ağırlığı",
            ],
            "Değer": [
                f"{ANALIZ_YILI} yıl", odeme_plani_adi, f"%{tufe_yuzde:.1f}",
                f"{dizel_fiyat:.2f} TL/L", f"{elektrik_fiyat:.2f} TL/kWh",
                f"%{ETA_SARJ*100:.0f}", f"{EF_GRID} kg CO₂/kWh [IEA 2023]",
                f"%{w_emisyon*100:.0f}", f"%{w_maliyet*100:.0f}",
            ],
        })
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Senaryo karşılaştırma
        st.markdown("#### 📋 Senaryo Karşılaştırma Özeti")
        ozet_df = pd.DataFrame({
            "Senaryo": [ETIKET[k] for k in ["S1","S2","S3"]],
            "Yıllık CO₂e (ton)": [f"{e['CO2e_ton']:,.1f}" for e in em_listesi],
            f"{ANALIZ_YILI}Y Toplam Maliyet (Milyar TL)": [
                f"{df_['toplam'].sum()/1e9:,.2f}" for df_ in [df_s1, df_s2, df_s3]],
            "AHP Skoru": [f"{v:.4f}" for v in ahp],
            "Öneri": ["✅ Önerilen" if ["S1","S2","S3"][i] == en_iyi else "" for i in range(3)],
        })
        st.dataframe(ozet_df, use_container_width=True, hide_index=True)

        # Aylık ham veri indirme
        st.markdown("#### 💾 Ham Veri İndir")
        col_d1, col_d2, col_d3 = st.columns(3)
        for col, kod, df_ in [(col_d1,"S1",df_s1),(col_d2,"S2",df_s2),(col_d3,"S3",df_s3)]:
            with col:
                csv = df_.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label=f"⬇ {kod} Aylık Veri (CSV)",
                    data=csv,
                    file_name=f"senaryo_{kod.lower()}_aylik.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
