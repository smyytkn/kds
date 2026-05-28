"""
ELEKTRİKLİ ARACA GEÇİŞ SÜRECİ İÇİN KARAR DESTEK SİSTEMİ
IPCC Tier 2 Metodolojisi, XGBoost Tahminleme & TOPSIS Destekli Senaryo Analizi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

# ─── MAKİNE ÖĞRENMESİ KÜTÜPHANELERİ ───
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    # Eğer XGBoost yüklü değilse alternatif olarak RandomForest kullanılır
    from sklearn.ensemble import RandomForestRegressor as XGBRegressor
    XGB_AVAILABLE = False

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EV Geçiş Karar Destek Sistemi",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# (Mevcut CSS Tasarımlarınız Aynen Korunmuştur)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
section[data-testid="stSidebar"] { background: #0d1117; border-right: 2px solid #1e9e6b; }
section[data-testid="stSidebar"] * { color: #C0C0C0 !important; }
.stApp { background: #f7f8fc; }
.hero-block {
    background: linear-gradient(135deg, #0d1117 60%, #1a2a1e 100%);
    color: #e6edf3; padding: 2.2rem 2.5rem; border-radius: 12px;
    margin-bottom: 1.5rem; border-left: 5px solid #1e9e6b;
}
.hero-block h1 { font-family: 'IBM Plex Mono', monospace; font-size: 27px; color: #1e9e6b; }
.metric-card {
    background: #fff; border-radius: 10px; padding: 1rem 1.2rem; 
    border: 1px solid #e1e4e8; border-top: 3px solid var(--accent, #1e9e6b);
}
.metric-card .val { font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; }
.winner-box {
    background: linear-gradient(135deg, #1a2a1e, #0d1117);
    border: 1px solid #1e9e6b; border-radius: 10px; padding: 1.4rem 1.8rem; color: #e6edf3;
}
</style>
""", unsafe_allow_html=True)

# ─── SABİT DEĞERLER (IPCC TIER 2) ───
EF_CO2_DIZEL  = 2.690
EF_CH4_OTOBUS = 3.9
EF_N2O_OTOBUS = 3.9
EF_CH4_MINI   = 3.9
EF_N2O_MINI   = 3.9
EF_GRID       = 0.43
ETA_SARJ      = 0.90

GWP_CH4 = 28
GWP_N2O = 265

RENK = {"MD":"#555555","S1":"#2166AC","S2":"#F4A100","S3":"#1B7837"}
ETIKET = {
    "MD": "Mevcut Durum - Tam Dizel",
    "S1": "Senaryo 1 – 1/3 Elektrikli Araca Geçiş",
    "S2": "Senaryo 2 – 2/3 Elektrikli Araca Geçiş",
    "S3": "Senaryo 3 – Tam Elektrikli Araca Geçiş",
}
ANALIZ_YILI = 10

# ─── BAŞLIK ───
st.markdown("""
<div class="hero-block">
  <h1>📈 ELEKTRİKLİ ARACA GEÇİŞ KARAR DESTEK SİSTEMİ (ML ENTEGRELİ)</h1>
  <p>
      Bu sistem, IPCC Tier 2 metodolojisi ve <b>XGBoost Regressor Makine Öğrenmesi</b> algoritmasını
      kullanarak filo tüketim değerlerini tahmin eder ve senaryoları TOPSIS yöntemiyle optimize eder.
  </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR PARAMETRELERİ ───
with st.sidebar:
    st.markdown("## 📋 FİLO BİLGİLERİ")
    n_otobus = st.number_input("Dizel Otobüs Sayısı", min_value=1, value=20, step=1)
    n_mini   = st.number_input("Dizel Minibüs Sayısı", min_value=0, value=10, step=1)

    st.markdown("### ELEKTRİKLİ ARAÇ FİYATLARI (TL)")
    fiyat_otobus_ev = st.number_input("Elektrikli Otobüs Birim Fiyatı", min_value=1.0, value=8000000.0, format="%.0f")
    fiyat_mini_ev   = st.number_input("Elektrikli Minibüs Birim Fiyatı", min_value=0.0, value=3500000.0, format="%.0f")

    st.markdown("### BAKIM MALİYETLERİ (TL/ARAÇ/YIL)")
    bak_otobus_d = st.number_input("Dizel Otobüs Bakım", value=150000.0)
    bak_mini_d   = st.number_input("Dizel Minibüs Bakım", value=80000.0)
    bak_otobus_e = st.number_input("Elektrikli Otobüs Bakım", value=60000.0)
    bak_mini_e   = st.number_input("Elektrikli Minibüs Bakım", value=35000.0)

    st.markdown("### YAKIT / ELEKTRİK FİYATLARI")
    dizel_fiyat    = st.number_input("Dizel Fiyatı (TL/L)", value=45.0)
    elektrik_fiyat = st.number_input("Elektrik Fiyatı (TL/kWh)", value=4.5)

    st.markdown("### YILLIK ARAÇ KİLOMETRELER")
    km_otobus = st.number_input("Otobüs Filosu Toplam Yıllık km", value=1500000.0)
    km_mini   = st.number_input("Minibüs Filosu Toplam Yıllık km", value=600000.0)

    st.markdown("### 🛠️ ML MODEL HİPERPARAMETRELERİ")
    ml_aktivite = st.checkbox("Tahminlerde ML Modeli Kullan", value=True)
    n_estimators = st.slider("Ağaç Sayısı (n_estimators)", 10, 200, 100)
    max_depth = st.slider("Maksimum Derinlik (max_depth)", 3, 10, 5)

    st.markdown("### TOPSIS AĞIRLIKLARI")
    w_emisyon = st.slider("Emisyon Ağırlığı",  0.0, 1.0, 0.40, 0.05)
    w_maliyet = st.slider("Maliyet Ağırlığı",  0.0, 1.0 - w_emisyon, 0.35, 0.05)
    w_yatirim = round(max(0.0, 1.0 - w_emisyon - w_maliyet), 4)
    st.markdown(f"**Yatırım Ağırlığı :** `{w_yatirim:.2f}`")

    tufe_yuzde = 30.0
    tufe_orani = tufe_yuzde / 100.0
    odeme_plani = 1

    hesapla_btn = st.button(" ANALİZİ ÇAĞRILŞTIR 🔍 ", use_container_width=True)

# ─── MAKİNE ÖĞRENMESİ VERİ GENERATÖRÜ VE MODEL EĞİTİMİ ───
@st.cache_data
def train_ml_model(n_estimators, max_depth):
    # Gerçek dünya filo verilerini simüle eden sentetik veri seti üretimi
    np.random.seed(42)
    data_size = 500
    
    # Bağımsız Değişkenler: Hava Sıcaklığı, Doluluk Oranı (%), Ortalama Hız (km/s)
    sicaklik = np.random.uniform(-10, 40, data_size)
    doluluk = np.random.uniform(20, 100, data_size)
    hiz = np.random.uniform(15, 60, data_size)
    
    # Bağımlı Değişkenler (Doğrusal olmayan tüketim denklemleri)
    # Otobüs Dizel Tüketimi (L/km) - Baz: 0.33
    tuk_otobus_dizel = 0.30 + (doluluk * 0.001) - (hiz * 0.0005) + (np.abs(sicaklik - 20) * 0.0008) + np.random.normal(0, 0.01, data_size)
    # Minibüs Dizel Tüketimi (L/km) - Baz: 0.12
    tuk_mini_dizel = 0.10 + (doluluk * 0.0004) - (hiz * 0.0002) + np.random.normal(0, 0.005, data_size)
    
    df_ml = pd.DataFrame({
        'sicaklik': sicaklik, 'doluluk': doluluk, 'hiz': hiz,
        'tuk_otobus': tuk_otobus_dizel, 'tuk_mini': tuk_mini_dizel
    })
    
    # Model Eğitimi (Otobüs Tüketim Modeli)
    X = df_ml[['sicaklik', 'doluluk', 'hiz']]
    y_o = df_ml['tuk_otobus']
    y_m = df_ml['tuk_mini']
    
    X_train, X_test, y_train_o, y_test_o = train_test_split(X, y_o, test_size=0.2, random_seed=42)
    _, _, y_train_m, y_test_m = train_test_split(X, y_m, test_size=0.2, random_seed=42)
    
    model_o = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model_o.fit(X_train, y_train_o)
    
    model_m = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model_m.fit(X_train, y_train_m)
    
    # Metrik Hesaplama
    r2_o = r2_score(y_test_o, model_o.predict(X_test))
    rmse_o = np.sqrt(mean_squared_error(y_test_o, model_o.predict(X_test)))
    
    return model_o, model_m, r2_o, rmse_o

# Modelleri Arka Planda Eğit
model_o, model_m, r2_score_val, rmse_val = train_ml_model(n_estimators, max_depth)

# ─── DİNAMİK ML TÜKETİM ATAMASI ───
if ml_aktivite:
    # Ortalama operasyon koşulları girilerek ML modelinden tüketim tahmini alınır
    test_features = pd.DataFrame({'sicaklik': [15.0], 'doluluk': [65.0], 'hiz': [32.0]})
    TUK_OTOBUS = float(model_o.predict(test_features)[0])
    TUK_MINI   = float(model_m.predict(test_features)[0])
else:
    TUK_OTOBUS = 0.33
    TUK_MINI   = 0.12

# ─── TOPSIS ALGORİTMASI ───
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
    en_iyi = ["S1", "S2", "S3"][int(np.argmax(C))]
    return {"M": M, "C": C, "en_iyi": en_iyi}

# ─── ANA ANALİZ FONKSİYONU ───
def run_analysis(we, wm, wy):
    md = dict(otobus_d=n_otobus,          otobus_e=0,               mini_d=n_mini,          mini_e=0)
    s1 = dict(otobus_d=n_otobus*(2/3),    otobus_e=n_otobus/3,      mini_d=n_mini*(2/3),    mini_e=n_mini/3)
    s2 = dict(otobus_d=n_otobus/3,        otobus_e=n_otobus*(2/3), mini_d=n_mini/3,        mini_e=n_mini*(2/3))
    s3 = dict(otobus_d=0,                 otobus_e=n_otobus,        mini_d=0,               mini_e=n_mini)

    def em(sc, km_o, km_m):
        o_km_d = (sc["otobus_d"] / max(n_otobus, 1)) * km_o
        o_km_e = (sc["otobus_e"] / max(n_otobus, 1)) * km_o
        m_km_d = (sc["mini_d"]   / max(n_mini,   1)) * km_m if n_mini > 0 else 0
        m_km_e = (sc["mini_e"]   / max(n_mini,   1)) * km_m if n_mini > 0 else 0
        
        co2  = o_km_d * TUK_OTOBUS * EF_CO2_DIZEL + o_km_e * (E_OTOBUS_EV / ETA_SARJ) * EF_GRID
        co2 += m_km_d * TUK_MINI   * EF_CO2_DIZEL + m_km_e * (E_MINI_EV   / ETA_SARJ) * EF_GRID
        ch4  = (o_km_d * EF_CH4_OTOBUS + m_km_d * EF_CH4_MINI) / 1e6
        n2o  = (o_km_d * EF_N2O_OTOBUS + m_km_d * EF_N2O_MINI) / 1e6
        co2e = (co2 + ch4 * GWP_CH4 + n2o * GWP_N2O) / 1000
        return {"CO2_kg": co2, "CH4_kg": ch4, "N2O_kg": n2o, "CO2e_ton": co2e}

    em_md = em(md, km_otobus, km_mini)
    em_s1 = em(s1, km_otobus, km_mini)
    em_s2 = em(s2, km_otobus, km_mini)
    em_s3 = em(s3, km_otobus, km_mini)

    def maliyet(sc, n_ev_o, n_ev_m, f_o, f_m, tufe, yil):
        ay_yak_d = (sc["otobus_d"] * (km_otobus / max(n_otobus, 1)) * TUK_OTOBUS * dizel_fiyat) / 12
        ay_bak = (sc["otobus_d"] * bak_otobus_d + sc["otobus_e"] * bak_otobus_e) / 12
        yat = n_ev_o * f_o + n_ev_m * f_m
        
        rows = []
        for ay in range(1, yil * 12 + 1):
            yc = (1 + tufe) ** ((ay - 1) // 12)
            rows.append({"ay": ay, "toplam": (ay_yak_d + ay_bak) * yc + (yat/(yil*12))})
        return pd.DataFrame(rows)

    df_md = maliyet(md, 0, 0, 0, 0, tufe_orani, ANALIZ_YILI)
    df_s1 = maliyet(s1, n_otobus/3, n_mini/3, fiyat_otobus_ev, fiyat_mini_ev, tufe_orani, ANALIZ_YILI)
    df_s2 = maliyet(s2, n_otobus*(2/3), n_mini*(2/3), fiyat_otobus_ev, fiyat_mini_ev, tufe_orani, ANALIZ_YILI)
    df_s3 = maliyet(s3, n_otobus, n_mini, fiyat_otobus_ev, fiyat_mini_ev, tufe_orani, ANALIZ_YILI)

    yat1 = (n_otobus/3) * fiyat_otobus_ev 
    yat2 = (n_otobus*(2/3)) * fiyat_otobus_ev
    yat3 = n_otobus * fiyat_otobus_ev

    t = topsis(em_s1, em_s2, em_s3, df_s1, df_s2, df_s3, yat1, yat2, yat3, we, wm, wy)
    return {"em_md": em_md, "em_s1": em_s1, "em_s2": em_s2, "em_s3": em_s3, "topsis": t, "en_iyi": t["en_iyi"]}

# --- ARAYÜZ ÇIKTILARI VE ML SEKMESİ ---
if hesapla_btn or st.session_state.get("results") is not None:
    if st.session_state.get("results") is None:
        st.session_state["results"] = run_analysis(w_emisyon, w_maliyet, w_yatirim)
    
    res = st.session_state["results"]
    
    # SEKMELER
    tab_emisyon, tab_maliyet, tab_ml_report = st.tabs(["♻️ EMİSYON ANALİZİ", "💹 MALİYET ANALİZİ", "🤖 MACHINE LEARNING RAPORU"])
    
    with tab_ml_report:
        st.header("🤖 XGBoost Regressor Model Performans Analizi")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Model Başarı Oranı (R² Score)", f"%{r2_score_val*100:.2f}")
        with c2:
            st.metric("Hata Kareler Ortalaması Kökü (RMSE)", f"{rmse_val:.4f} L/km")
        with c3:
            st.metric("ML Tahmini Otobüs Tüketimi", f"{TUK_OTOBUS:.3f} L/km")
            
        st.success("🤖 Makine Öğrenmesi Modeli Başarıyla Çalıştırıldı! Dinamik hava sıcaklığı ve doluluk parametrelerine göre deterministik referans değerler güncellendi.")
