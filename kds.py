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
EF_CO2_DIZEL   = 2.690   # kg/L
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

RENK = {
    "MD": "#555555",
    "S1": "#2166AC",
    "S2": "#F4A100",
    "S3": "#1B7837",
}
ETIKET = {
    "MD": "Mevcut Durum (Tam Dizel)",
    "S1": "Senaryo 1 – 1/3 EV Geçişi",
    "S2": "Senaryo 2 – 2/3 EV Geçişi",
    "S3": "Senaryo 3 – Tam EV Geçişi",
}

# ─────────────────────────────────────────────
#  BAŞLIK
# ─────────────────────────────────────────────
st.markdown("""
<style>
.hero-block h1 {
    font-size: 27px;
    margin-bottom: 10px;
}

.hero-block p {
    font-size: 18px;
    line-height: 1.6;
    margin-top: 0;
}
</style>

<div class="hero-block">
  <h1> 📈 ELEKTRİKLİ ARACA GEÇİŞ KARAR DESTEK SİSTEMİ</h1>
  <p>
    Bu uygulama, dizel araç filolarının elektrikli araçlara geçiş sürecinde emisyon ve maliyet parametrelerine dayalı senaryo analizleri gerçekleştirerek geçiş kararlarının optimizasyonunu desteklemektedir.
  </p>
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

    st.markdown("### ARAÇ FİYATLARI (TL)")
    fiyat_otobüs_ev  = st.number_input("Elektrikli Otobüs Birim Fiyatı (TL)", min_value=1.0,
                                        value=8_000_000.0, step=100_000.0, format="%.0f")
    fiyat_minibüs_ev = st.number_input("Elektrikli Minibüs Birim Fiyatı (TL)", min_value=0.0,
                                        value=3_500_000.0, step=100_000.0, format="%.0f")

    st.markdown("### BAKIM MALİYETLERİ (TL/ARAÇ/YIL)")
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

    ANALIZ_YILI = 20

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

# ─────────────────────────────────────────────
#  HESAPLAMALAR
# ─────────────────────────────────────────────

def run_analysis(w_emisyon, w_maliyet):
    # Senaryo tanımları (Yazım hatası düzeltildi: minibüs_ev=n_minibüs_mevcut*(2/3))
    md = dict(otobüs_dizel=n_otobüs_mevcut, otobüs_ev=0,
               minibüs_dizel=n_minibüs_mevcut, minibüs_ev=0)
    s1 = dict(otobüs_dizel=n_otobüs_mevcut*(2/3), otobüs_ev=n_otobüs_mevcut/3,
               minibüs_dizel=n_minibüs_mevcut*(2/3), minibüs_ev=n_minibüs_mevcut/3)
    s2 = dict(otobüs_dizel=n_otobüs_mevcut/3, otobüs_ev=n_otobüs_mevcut*(2/3),
               minibüs_dizel=n_minibüs_mevcut/3, minibüs_ev=n_minibüs_mevcut*(2/3))
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

    em_md = emisyon_hesapla(md, km_otobüs_yillik, km_minibüs_yillik)
    em_s1 = emisyon_hesapla(s1, km_otobüs_yillik, km_minibüs_yillik)
    em_s2 = emisyon_hesapla(s2, km_otobüs_yillik, km_minibüs_yillik)
    em_s3 = emisyon_hesapla(s3, km_otobüs_yillik, km_minibüs_yillik)

    def maliyet_serileri(senaryo, n_arac_ev_oto, n_arac_ev_mini,
                         fiyat_oto_ev, fiyat_mini_ev, tufe, yil, odeme_plan):
        
        # Sabit aylık yakıt ve bakım (başlangıç yılı)
        ayl_yakıt_dizel = (
            senaryo["otobüs_dizel"]  * (km_otobüs_yillik  / n_otobüs_mevcut  if n_otobüs_mevcut  else 0) * TUK_OTOBÜS_DIZEL  * dizel_fiyat +
            senaryo["minibüs_dizel"] * (km_minibüs_yillik / n_minibüs_mevcut if n_minibüs_mevcut else 0) * TUK_MINİBÜS_DIZEL * dizel_fiyat
        ) / 12
        ayl_yakıt_ev = (
            senaryo["otobüs_ev"]  * (km_otobüs_yillik  / n_otobüs_mevcut  if n_otobüs_mevcut  else 0) * (E_OTOBÜS_EV  / ETA_SARJ) * elektrik_fiyat +
            senaryo["minibüs_ev"] * (km_minibüs_yillik / n_minibüs_mevcut if n_minibüs_mevcut else 0) * (E_MINİBÜS_EV / ETA_SARJ) * elektrik_fiyat
        ) / 12
        ayl_bakım_dizel = (
            senaryo["otobüs_dizel"]  * bakim_otobüs_dizel  +
            senaryo["minibüs_dizel"] * bakim_minibüs_dizel
        ) / 12
        ayl_bakım_ev = (
            senaryo["otobüs_ev"]     * bakim_otobüs_ev     +
            senaryo["minibüs_ev"]    * bakim_minibüs_ev
        ) / 12
        
        # Toplam yatırım (borç)
        yatirim_toplam = n_arac_ev_oto * fiyat_oto_ev + n_arac_ev_mini * fiyat_mini_ev
        
        # Ödeme planı hesaplama
        if odeme_plan == 1:  # Sabit Ödeme
            toplam_giderler = 0
            for yn in range(yil):
                yc = (1 + tufe) ** yn
                yillik_yakıt_ev = ayl_yakıt_ev * 12 * yc
                yillik_bakım_ev = ayl_bakım_ev * 12 * yc
                toplam_giderler += yillik_yakıt_ev + yillik_bakım_ev
            
            aylik_odeme_sabit = (yatirim_toplam + toplam_giderler) / (yil * 12) if yil > 0 else 0
            
        else:  # TÜFE Oranında Artan Ödeme
            toplam_giderler = 0
            for yn in range(yil):
                yc = (1 + tufe) ** yn
                yillik_yakıt_ev = ayl_yakıt_ev * 12 * yc
                yillik_bakım_ev = ayl_bakım_ev * 12 * yc
                toplam_giderler += yillik_yakıt_ev + yillik_bakım_ev
            
            odeme_toplam = yatirim_toplam + toplam_giderler
            aylık_odeme_tüfe_base = odeme_toplam / (yil * 12) if yil > 0 else 0

        kayitlar = []
        kalan_borç = yatirim_toplam
        
        for ay in range(1, yil * 12 + 1):
            yn = (ay - 1) // 12  
            yc = (1 + tufe) ** yn  
            
            yakıt_dizel = ayl_yakıt_dizel * yc
            yakıt_ev = ayl_yakıt_ev * yc
            bakım_dizel = ayl_bakım_dizel * yc
            bakım_ev = ayl_bakım_ev * yc
            
            if odeme_plan == 1:  
                odeme = aylik_odeme_sabit
            else:  
                odeme = aylık_odeme_tüfe_base * yc
            
            kalan_borç = max(0, kalan_borç - odeme)
            
            if n_arac_ev_oto == 0 and n_arac_ev_mini == 0:  
                scenario_type = "MD"
                toplam_yakıt = yakıt_dizel
                toplam_bakım = bakım_dizel
                ödeme_miktarı = 0  
                kalan_borç = 0
                toplam_maliyet = toplam_yakıt + toplam_bakım
            else:
                scenario_type = "EV"
                toplam_yakıt = yakıt_dizel + yakıt_ev
                toplam_bakım = bakım_dizel + bakım_ev
                ödeme_miktarı = odeme
                toplam_maliyet = toplam_yakıt + toplam_bakım + ödeme_miktarı
            
            kayitlar.append({
                "ay": ay,
                "yil": yn + 1,
                "yakıt_dizel": yakıt_dizel,
                "yakıt_ev": yakıt_ev,
                "bakım_dizel": bakım_dizel,
                "bakım_ev": bakım_ev,
                "toplam_yakıt": toplam_yakıt,
                "toplam_bakım": toplam_bakım,
                "odeme": ödeme_miktarı,
                "kalan_borc": kalan_borç,
                "toplam": toplam_maliyet,
                "scenario_type": scenario_type
            })
        
        return pd.DataFrame(kayitlar)

    df_md = maliyet_serileri(md, 0, 0, 0, 0, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s1 = maliyet_serileri(s1, n_otobüs_mevcut/3, n_minibüs_mevcut/3, fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s2 = maliyet_serileri(s2, n_otobüs_mevcut*(2/3), n_minibüs_mevcut*(2/3), fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)
    df_s3 = maliyet_serileri(s3, n_otobüs_mevcut, n_minibüs_mevcut, fiyat_otobüs_ev, fiyat_minibüs_ev, tufe_orani, ANALIZ_YILI, odeme_plani)

    # AHP
    em_d  = np.array([em_s1["CO2e_ton"], em_s2["CO2e_ton"], em_s3["CO2e_ton"]])
    mal_d = np.array([df_s1["toplam"].sum(), df_s2["toplam"].sum(), df_s3["toplam"].sum()])
    def norm_min(v):
        inv = 1.0 / (v + 1e-12); return inv / inv.sum()
    em_norm  = norm_min(em_d)
    mal_norm = norm_min(mal_d)
    ahp      = w_emisyon * em_norm + w_maliyet * mal_norm
    en_iyi   = ["S1","S2","S3"][np.argmax(ahp)]

    return {
        "s1": s1, "s2": s2, "s3": s3,
        "em_md": em_md, "em_s1": em_s1, "em_s2": em_s2, "em_s3": em_s3,
        "df_md": df_md, "df_s1": df_s1, "df_s2": df_s2, "df_s3": df_s3,
        "em_norm": em_norm, "mal_norm": mal_norm, "ahp": ahp, "en_iyi": en_iyi,
    }

# ─────────────────────────────────────────────
#  BAŞLANGIÇ YA DA HESAP SONRASI GÖSTERİM
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None

if hesapla_btn:
    with st.spinner("Analiz çalışıyor…"):
        st.session_state["results"] = run_analysis(w_emisyon, w_maliyet)

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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="background:#1a1f2e; border-left:3px solid #555555; border-radius:8px; padding:0.7rem 0.9rem; margin-bottom:0.5rem;">
            <div style="color:#aaa; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;">MD</div>
            <div style="color:#e6edf3; font-size:0.82rem; font-weight:600;">🚌 Mevcut Durum</div>
            <div style="color:#8b949e; font-size:0.75rem; margin-top:2px;">Filo tamamen dizel araçlardan oluşur. Referans senaryo.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#1a1f2e; border-left:3px solid #2166AC; border-radius:8px; padding:0.7rem 0.9rem; margin-bottom:0.5rem;">
            <div style="color:#2166AC; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;">S1</div>
            <div style="color:#e6edf3; font-size:0.82rem; font-weight:600;"> 1/3 EV Geçişi</div>
            <div style="color:#8b949e; font-size:0.75rem; margin-top:2px;">Filonun <b style="color:#2166AC">%33</b>'ü elektrikli araca dönüştürülür.</div>
            <div style="margin-top:5px;">
              <span style="background:#2166AC22; color:#2166AC; border-radius:3px; font-size:0.68rem; padding:1px 6px;">Düşük Yatırım</span>
              <span style="background:#2166AC22; color:#2166AC; border-radius:3px; font-size:0.68rem; padding:1px 6px; margin-left:3px;">Kademeli Geçiş</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#1a1f2e; border-left:3px solid #F4A100; border-radius:8px; padding:0.7rem 0.9rem; margin-bottom:0.5rem;">
            <div style="color:#F4A100; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;">S2</div>
            <div style="color:#e6edf3; font-size:0.82rem; font-weight:600;"> 2/3 EV Geçişi</div>
            <div style="color:#8b949e; font-size:0.75rem; margin-top:2px;">Filonun <b style="color:#F4A100">%67</b>'si elektrikli araca dönüştürülür.</div>
            <div style="margin-top:5px;">
              <span style="background:#F4A10022; color:#F4A100; border-radius:3px; font-size:0.68rem; padding:1px 6px;">Orta Yatırım</span>
              <span style="background:#F4A10022; color:#F4A100; border-radius:3px; font-size:0.68rem; padding:1px 6px; margin-left:3px;">Dengeli Plan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="background:#1a1f2e; border-left:3px solid #1B7837; border-radius:8px; padding:0.7rem 0.9rem; margin-bottom:0.5rem;">
            <div style="color:#1B7837; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;">S3</div>
            <div style="color:#e6edf3; font-size:0.82rem; font-weight:600;"> Tam EV Geçişi</div>
            <div style="color:#8b949e; font-size:0.75rem; margin-top:2px;">Filonun <b style="color:#1B7837">%100</b>'ü elektrikli araca dönüştürülür.</div>
            <div style="margin-top:5px;">
              <span style="background:#1B783722; color:#1B7837; border-radius:3px; font-size:0.68rem; padding:1px 6px;">Yüksek Yatırım</span>
              <span style="background:#1B783722; color:#1B7837; border-radius:3px; font-size:0.68rem; padding:1px 6px; margin-left:3px;">Max. Emisyon Azalımı</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""<hr style="border-color:#2a3a2e; margin:0.8rem 0;">""", unsafe_allow_html=True)

else:
    em_md, em_s1, em_s2, em_s3 = res["em_md"], res["em_s1"], res["em_s2"], res["em_s3"]
    df_md, df_s1, df_s2, df_s3 = res["df_md"], res["df_s1"], res["df_s2"], res["df_s3"]
    ahp, em_norm, mal_norm, en_iyi = res["ahp"], res["em_norm"], res["mal_norm"], res["en_iyi"]

    em_listesi = [em_s1, em_s2, em_s3]
    df_listesi = [df_s1, df_s2, df_s3]

    # Özet metrik kartları
    col1, col2, col3, col4 = st.columns(4)
    s1_azalma = (1 - em_s1["CO2e_ton"] / em_md["CO2e_ton"]) * 100 if em_md["CO2e_ton"] > 0 else 0
    s2_azalma = (1 - em_s2["CO2e_ton"] / em_md["CO2e_ton"]) * 100 if em_md["CO2e_ton"] > 0 else 0
    s3_azalma = (1 - em_s3["CO2e_ton"] / em_md["CO2e_ton"]) * 100 if em_md["CO2e_ton"] > 0 else 0
    with col1:
        st.markdown(f"""<div class="metric-card" style="--accent:#555555">
          <div class="lbl">Mevcut Durum Yıllık Emisyon</div>
          <div class="val">{em_md['CO2e_ton']:,.0f}</div>
          <div class="lbl">ton CO₂e/yıl</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card" style="--accent:#2166AC">
          <div class="lbl">Senaryo 1 – 1/3 EV Geçişi Emisyon Azalması</div>
          <div class="val">▼{s1_azalma:.1f}%</div>
          <div class="lbl">MEVCUT DURUMA KIYASLA</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card" style="--accent:#F4A100">
          <div class="lbl">Senaryo 2 – 2/3 EV Geçişi Emisyon Azalması</div>
          <div class="val">▼{s2_azalma:.1f}%</div>
          <div class="lbl">MEVCUT DURUMA KIYASLA </div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card" style="--accent:#1B7837">
          <div class="lbl">Senaryo 3 – 3/3 EV Geçişi Emisyon Azalması</div>
          <div class="val">▼{s3_azalma:.1f}%</div>
          <div class="lbl">MEVCUT DURUMA KIYASLA</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Kazanan Öneri Kutusu
    st.markdown(f"""
    <div class="winner-box">
        <div class="wlbl">🏆 ÖNERİLEN OPTİMAL SENARYO </div>
        <div class="wval">{ETIKET[en_iyi]}</div>
        <div style="font-size:0.85rem; color:#8b949e; margin-top:6px;">
            Emisyon azaltımı ve toplam maliyet kriterleri %50-%50 ağırlıklandırılarak yapılan analitik hiyerarşi süreci sonucunda en dengeli geçiş stratejisi seçilmiştir.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Ana sekmeler
    tab_emisyon, tab_maliyet, tab_basabas, tab_tablo = st.tabs([
        "♻️ EMİSYON ANALİZİ",
        "💹 MALİYET ANALİZİ",
        "🎯 BAŞABAŞ ANALİZİ",
        "📊 DETAY TABLOLAR",
    ])

    # ──────────────────────────────────────────
    #  EMİSYON SEKMESİ
    # ──────────────────────────────────────────
    with tab_emisyon:
        st.subheader(" Senaryo Bazlı Yıllık Emisyon Karşılaştırması")

        import matplotlib as mpl
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['axes.unicode_minus'] = False

        senaryolar = ["MD", "S1", "S2", "S3"]
        renkler    = [RENK[k] for k in senaryolar]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle(
            "IPCC Tier 2 – Senaryo Bazlı Yıllık Emisyon Karşılaştırması\nKarabük UlaşımAŞ Toplu Taşıma Filosu",
            fontsize=12, fontweight="bold"
        )
        em_tum_listesi = [em_md, em_s1, em_s2, em_s3]
        for ax, key, birim, fmt, bolucu in [
            (axes[0,0], "CO2_kg",   "CO₂ (ton/yıl)",   ",.0f", 1000),
            (axes[0,1], "CH4_kg",   "CH₄ (kg/yıl)",    ",.3f", 1),
            (axes[1,0], "N2O_kg",   "N₂O (kg/yıl)",    ",.3f", 1),
            (axes[1,1], "CO2e_ton", "CO₂e (ton/yıl)", ",.1f", 1),
        ]:
            vals = [e[key] / bolucu for e in em_tum_listesi]
            bars = ax.bar(["MD","S1","S2","S3"], vals, color=renkler, width=0.5, edgecolor="white", linewidth=1.2)
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

        st.markdown("#### Emisyon Karşılaştırma Tablosu")
        tablo_data = {
            "Gösterge": ["CO₂ (ton/yıl)", "CH₄ (kg/yıl)", "N₂O (kg/yıl)", "CO₂e (ton/yıl)"],
            "MD – Mevcut Durum": [
                f"{em_md['CO2_kg']/1000:,.1f}", f"{em_md['CH4_kg']:,.3f}",
                f"{em_md['N2O_kg']:,.3f}", f"{em_md['CO2e_ton']:,.2f}"],
            "S1 – 1/3 EV": [
                f"{em_s1['CO2_kg']/1000:,.1f}", f"{em_s1['CH4_kg']:,.3f}",
                f"{em_s1['N2O_kg']:,.3f}", f"{em_s1['CO2e_ton']:,.2f}"],
            "S1 Azalma": [
                f"▼%{(1-em_s1['CO2_kg']/em_md['CO2_kg'])*100:.1f}" if em_md['CO2_kg']>0 else "-",
                f"▼%{(1-em_s1['CH4_kg']/em_md['CH4_kg'])*100:.1f}" if em_md['CH4_kg']>0 else "-",
                f"▼%{(1-em_s1['N2O_kg']/em_md['N2O_kg'])*100:.1f}" if em_md['N2O_kg']>0 else "-",
                f"▼%{(1-em_s1['CO2e_ton']/em_md['CO2e_ton'])*100:.1f}" if em_md['CO2e_ton']>0 else "-"],
            "S2 – 2/3 EV": [
                f"{em_s2['CO2_kg']/1000:,.1f}", f"{em_s2['CH4_kg']:,.3f}",
                f"{em_s2['N2O_kg']:,.3f}", f"{em_s2['CO2e_ton']:,.2f}"],
            "S2 Azalma": [
                f"▼%{(1-em_s2['CO2_kg']/em_md['CO2_kg'])*100:.1f}" if em_md['CO2_kg']>0 else "-",
                f"▼%{(1-em_s2['CH4_kg']/em_md['CH4_kg'])*100:.1f}" if em_md['CH4_kg']>0 else "-",
                f"▼%{(1-em_s2['N2O_kg']/em_md['N2O_kg'])*100:.1f}" if em_md['N2O_kg']>0 else "-",
                f"▼%{(1-em_s2['CO2e_ton']/em_md['CO2e_ton'])*100:.1f}" if em_md['CO2e_ton']>0 else "-"],
            "S3 – 3/3 EV": [
                f"{em_s3['CO2_kg']/1000:,.1f}", f"{em_s3['CH4_kg']:,.3f}",
                f"{em_s3['N2O_kg']:,.3f}", f"{em_s3['CO2e_ton']:,.2f}"],
            "S3 Azalma": [
                f"▼%{(1-em_s3['CO2_kg']/em_md['CO2_kg'])*100:.1f}" if em_md['CO2_kg']>0 else "-",
                f"▼%{(1-em_s3['CH4_kg']/em_md['CH4_kg'])*100:.1f}" if em_md['CH4_kg']>0 else "-",
                f"▼%{(1-em_s3['N2O_kg']/em_md['N2O_kg'])*100:.1f}" if em_md['N2O_kg']>0 else "-",
                f"▼%{(1-em_s3['CO2e_ton']/em_md['CO2e_ton'])*100:.1f}" if em_md['CO2e_ton']>0 else "-"],
        }
        st.dataframe(pd.DataFrame(tablo_data), use_container_width=True, hide_index=True)
        st.info("TABLODAKİ AZALMA DEĞERLERİ MEVDUT DURUMA GÖRE KIYASLANMIŞTIR.")

    # ──────────────────────────────────────────
    #  MALİYET SEKMESİ
    # ──────────────────────────────────────────
    with tab_maliyet:
        st.subheader("💹 Senaryo Bazlı Aylık Maliyet Analizi")

        fig_cost, ax_cost = plt.subplots(figsize=(14, 6))
        scenarios = ["MD", "S1", "S2", "S3"]
        scenario_dfs = [df_md, df_s1, df_s2, df_s3]
        
        for df_sc, kod in zip(scenario_dfs, scenarios):
            ax_cost.plot(df_sc["ay"], df_sc["toplam"]/1e6, linewidth=2.5, 
                        color=RENK[kod], label=f"{ETIKET[kod]}")

        ax_cost.set_title(f"Zamana Bağlı Aylık Toplam Maliyet Trendleri ({ANALIZ_YILI} Yıl)", 
                          fontweight="bold", fontsize=12)
        ax_cost.set_xlabel("Ay", fontweight="bold")
        ax_cost.set_ylabel("Aylık Toplam Maliyet (Milyon TL)", fontweight="bold")
        ax_cost.legend(loc="upper left", fontsize=9)
        ax_cost.grid(True, linestyle=":", alpha=0.5)
        ax_cost.xaxis.set_major_locator(mticker.MultipleLocator(12))
        
        for y in range(1, ANALIZ_YILI + 1):
            ax_cost.axvline(y * 12, color="gray", lw=0.4, alpha=0.25, linestyle="-")

        plt.tight_layout()
        st.pyplot(fig_cost)
        plt.close(fig_cost)

        st.markdown("---")
        st.subheader(f"Senaryo Bazlı Detaylı Aylık Maliyet Bölümü – {ANALIZ_YILI} Yıl")

        for df_, kod in [(df_s1,"S1"), (df_s2,"S2"), (df_s3,"S3")]:
            with st.expander(f"📊 {ETIKET[kod]}", expanded=(kod=="S2")):
                col_g, col_t = st.columns([2, 1])
                with col_g:
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    renk = RENK[kod]
                    
                    # Stacked area grafiğinin tamamlanması
                    ax2.fill_between(df_["ay"], 0, df_["toplam_yakıt"]/1e6, 
                                    alpha=0.5, color=renk, label="Yakıt (Toplam)")
                    ax2.fill_between(df_["ay"], df_["toplam_yakıt"]/1e6, 
                                    (df_["toplam_yakıt"]+df_["toplam_bakım"])/1e6,
                                    alpha=0.5, color="gray", label="Bakım (Toplam)")
                    ax2.fill_between(df_["ay"], (df_["toplam_yakıt"]+df_["toplam_bakım"])/1e6,
                                    df_["toplam"]/1e6,
                                    alpha=0.3, color="red", label="Kredi/Yatırım Ödemesi")

                    ax2.set_title(f"{ETIKET[kod]} - Aylık Maliyet Dağılımı", fontweight="bold")
                    ax2.set_xlabel("Ay")
                    ax2.set_ylabel("Milyon TL")
                    ax2.legend(loc="upper left")
                    ax2.grid(True, linestyle=":", alpha=0.3)
                    st.pyplot(fig2)
                    plt.close(fig2)
                
                with col_t:
                    st.markdown("##### Finansal Özet Gelişimi")
                    st.write(f"**Toplam Operasyonel Ödeme (20 Yıl):** {df_['toplam'].sum()/1e6:,.1f} Milyon TL")
                    st.write(f"**Toplam Yakıt Gideri:** {df_['toplam_yakıt'].sum()/1e6:,.1f} Milyon TL")
                    st.write(f"**Toplam Bakım Gideri:** {df_['toplam_bakım'].sum()/1e6:,.1f} Milyon TL")
                    st.write(f"**Yatırım/Finansman Gideri:** {df_['odeme'].sum()/1e6:,.1f} Milyon TL")

    # ──────────────────────────────────────────
    #  BAŞABAŞ SEKMESİ
    # ──────────────────────────────────────────
    with tab_basabas:
        st.subheader("🎯 Kümülatif Maliyet ve Başabaş (Break-Even) Noktası Analizi")
        st.markdown("""
        Bu grafik, elektrikli araç dönüşüm yatırımının (başlangıç borcu ve finansman maliyetleri dahil) 
        dizel yakıt ve yüksek bakım tasarrufları sayesinde **Mevcut Durum'un kümülatif maliyetini hangi noktada kestiğini** gösterir.
        """)

        fig_be, ax_be = plt.subplots(figsize=(14, 6.5))
        
        cum_md = df_md["toplam"].cumsum() / 1e6
        ax_be.plot(df_md["ay"], cum_md, color=RENK["MD"], linewidth=3, label=f"Kümülatif {ETIKET['MD']}")

        for kod in ["S1", "S2", "S3"]:
            df_sc = res[f"df_{kod.lower()}"]
            cum_sc = df_sc["toplam"].cumsum() / 1e6
            ax_be.plot(df_sc["ay"], cum_sc, color=RENK[kod], linewidth=2.5, label=f"Kümülatif {ETIKET[kod]}")

            diff = cum_sc - cum_md
            cross_idx = np.where(diff < 0)[0]
            if len(cross_idx) > 0:
                break_even_month = df_sc["ay"].iloc[cross_idx[0]]
                break_even_value = cum_sc.iloc[cross_idx[0]]
                
                ax_be.plot(break_even_month, break_even_value, 'o', color=RENK[kod], markersize=10)
                ax_be.annotate(
                    f"{kod} Başabaş: {break_even_month}. Ay\n({break_even_month/12:.1f} Yıl)",
                    xy=(break_even_month, break_even_value),
                    xytext=(break_even_month - 15, break_even_value + max(cum_md)*0.08),
                    arrowprops=dict(arrowstyle="->", color=RENK[kod], lw=1.2),
                    fontsize=9, fontweight="bold", color=RENK[kod],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RENK[kod], alpha=0.8)
                )

        ax_be.set_title("Kümülatif Toplam Maliyet Gelişimi ve Yatırım Geri Dönüş Süreleri", fontweight="bold", fontsize=12)
        ax_be.set_xlabel("Proje Ömrü (Ay)", fontweight="bold")
        ax_be.set_ylabel("Kümülatif Toplam Maliyet (Milyon TL)", fontweight="bold")
        ax_be.legend(loc="upper left")
        ax_be.grid(True, linestyle=":", alpha=0.5)
        ax_be.xaxis.set_major_locator(mticker.MultipleLocator(12))
        
        plt.tight_layout()
        st.pyplot(fig_be)
        plt.close(fig_be)

    # ──────────────────────────────────────────
    #  DETAY TABLOLAR SEKMESİ
    # ──────────────────────────────────────────
    with tab_tablo:
        st.subheader("📊 Aylık Detay Veri Tabloları")
        st.markdown("Simülasyon çıktılarını Excel formatına aktarmak veya satır satır incelemek için ilgili senaryoyu seçin:")
        
        secilen_tablo = st.selectbox(
            "Görüntülenecek Senaryo Çıktısı",
            options=["MD", "S1", "S2", "S3"],
            format_func=lambda x: ETIKET[x]
        )
        
        df_target = res[f"df_{secilen_tablo.lower()}"]
        df_display = df_target.copy()
        df_display.columns = [
            "Ay", "Yıl", "Dizel Yakıt Gideri (TL)", "EV Enerji Gideri (TL)", 
            "Dizel Bakım Gideri (TL)", "EV Bakım Gideri (TL)", "Toplam Yakıt Gideri (TL)", 
            "Toplam Bakım Gideri (TL)", "Kredi/Yatırım Ödemesi (TL)", "Kalan Finansman Borcu (TL)", 
            "Aylık Toplam Maliyet (TL)", "Senaryo Tipi"
        ]
        
        st.dataframe(
            df_display.style.format({
                "Dizel Yakıt Gideri (TL)": "{:,.0f}",
                "EV Enerji Gideri (TL)": "{:,.0f}",
                "Dizel Bakım Gideri (TL)": "{:,.0f}",
                "EV Bakım Gideri (TL)": "{:,.0f}",
                "Toplam Yakıt Gideri (TL)": "{:,.0f}",
                "Toplam Bakım Gideri (TL)": "{:,.0f}",
                "Kredi/Yatırım Ödemesi (TL)": "{:,.0f}",
                "Kalan Finansman Borcu (TL)": "{:,.0f}",
                "Aylık Toplam Maliyet (TL)": "{:,.0f}"
            }),
            use_container_width=True,
            hide_index=True
        )
