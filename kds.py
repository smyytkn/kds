import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# =========================================================
# SAYFA AYARLARI
# =========================================================

st.set_page_config(
    page_title="EV Geçiş Karar Destek Sistemi",
    page_icon="⚡",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp{
    background-color:#f5f7fb;
}

.block-container{
    padding-top:2rem;
}

.metric-box{
    background:white;
    padding:1rem;
    border-radius:12px;
    border-left:5px solid #1B7837;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.big-title{
    background:linear-gradient(135deg,#0d1117,#1b7837);
    padding:2rem;
    border-radius:14px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SABİTLER
# =========================================================

EF_CO2_DIZEL = 2.69
EF_GRID = 0.43

TUK_OTOBUS_DIZEL = 0.33
TUK_MINIBUS_DIZEL = 0.12

E_OTOBUS_EV = 0.18
E_MINIBUS_EV = 0.12

EF_CH4 = 3.9
EF_N2O = 3.9

GWP_CH4 = 28
GWP_N2O = 265

RENKLER = {
    "MD":"#555555",
    "S1":"#2166AC",
    "S2":"#F4A100",
    "S3":"#1B7837"
}

# =========================================================
# BAŞLIK
# =========================================================

st.markdown("""
<div class='big-title'>
<h1>⚡ ELEKTRİKLİ ARACA GEÇİŞ KARAR DESTEK SİSTEMİ</h1>
<p>
IPCC Tier 2 metodolojisi ile elektrikli araç dönüşüm senaryo analizi
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📋 GİRİŞ PARAMETRELERİ")

    n_otobus = st.number_input(
        "Dizel Otobüs Sayısı",
        min_value=1,
        value=20
    )

    n_minibus = st.number_input(
        "Dizel Minibüs Sayısı",
        min_value=0,
        value=10
    )

    st.subheader("Araç Fiyatları")

    fiyat_otobus_ev = st.number_input(
        "Elektrikli Otobüs Fiyatı (TL)",
        value=8000000.0
    )

    fiyat_minibus_ev = st.number_input(
        "Elektrikli Minibüs Fiyatı (TL)",
        value=3500000.0
    )

    st.subheader("Bakım Maliyetleri")

    bakim_otobus_dizel = st.number_input(
        "Dizel Otobüs Bakım",
        value=150000.0
    )

    bakim_minibus_dizel = st.number_input(
        "Dizel Minibüs Bakım",
        value=80000.0
    )

    bakim_otobus_ev = st.number_input(
        "EV Otobüs Bakım",
        value=60000.0
    )

    bakim_minibus_ev = st.number_input(
        "EV Minibüs Bakım",
        value=35000.0
    )

    st.subheader("Yakıt / Elektrik")

    dizel_fiyat = st.number_input(
        "Dizel Fiyatı (TL/L)",
        value=45.0
    )

    elektrik_fiyat = st.number_input(
        "Elektrik Fiyatı (TL/kWh)",
        value=4.5
    )

    st.subheader("Yıllık Kilometre")

    km_otobus = st.number_input(
        "Otobüs Yıllık KM",
        value=1500000.0
    )

    km_minibus = st.number_input(
        "Minibüs Yıllık KM",
        value=600000.0
    )

    st.subheader("Enflasyon")

    tufe = st.slider(
        "Yıllık TÜFE (%)",
        0,
        100,
        30
    )

    analiz_yili = st.slider(
        "Analiz Süresi (Yıl)",
        1,
        30,
        20
    )

    hesapla = st.button("🔍 ANALİZİ ÇALIŞTIR")

# =========================================================
# FONKSİYONLAR
# =========================================================

def emisyon_hesapla(senaryo):

    oto_dizel = senaryo["otobus_dizel"]
    oto_ev = senaryo["otobus_ev"]

    mini_dizel = senaryo["minibus_dizel"]
    mini_ev = senaryo["minibus_ev"]

    co2_dizel = (
        oto_dizel * km_otobus * TUK_OTOBUS_DIZEL * EF_CO2_DIZEL
    )

    co2_dizel += (
        mini_dizel * km_minibus * TUK_MINIBUS_DIZEL * EF_CO2_DIZEL
    )

    co2_ev = (
        oto_ev * km_otobus * E_OTOBUS_EV * EF_GRID
    )

    co2_ev += (
        mini_ev * km_minibus * E_MINIBUS_EV * EF_GRID
    )

    toplam_co2 = co2_dizel + co2_ev

    ch4 = (
        (oto_dizel * km_otobus * EF_CH4)
        +
        (mini_dizel * km_minibus * EF_CH4)
    ) / 1e6

    n2o = (
        (oto_dizel * km_otobus * EF_N2O)
        +
        (mini_dizel * km_minibus * EF_N2O)
    ) / 1e6

    co2e = (
        toplam_co2 +
        ch4 * GWP_CH4 +
        n2o * GWP_N2O
    ) / 1000

    return {
        "CO2": toplam_co2 / 1000,
        "CH4": ch4,
        "N2O": n2o,
        "CO2e": co2e
    }

def maliyet_hesapla(senaryo):

    yillar = np.arange(1, analiz_yili + 1)

    yakit_liste = []
    toplam_liste = []

    yatirim = (
        senaryo["otobus_ev"] * fiyat_otobus_ev
        +
        senaryo["minibus_ev"] * fiyat_minibus_ev
    )

    for yil in yillar:

        katsayi = (1 + tufe/100) ** (yil-1)

        dizel_yakit = (
            senaryo["otobus_dizel"]
            *
            (km_otobus/n_otobus)
            *
            TUK_OTOBUS_DIZEL
            *
            dizel_fiyat
        )

        dizel_yakit += (
            senaryo["minibus_dizel"]
            *
            (km_minibus/max(n_minibus,1))
            *
            TUK_MINIBUS_DIZEL
            *
            dizel_fiyat
        )

        elektrik = (
            senaryo["otobus_ev"]
            *
            (km_otobus/n_otobus)
            *
            E_OTOBUS_EV
            *
            elektrik_fiyat
        )

        elektrik += (
            senaryo["minibus_ev"]
            *
            (km_minibus/max(n_minibus,1))
            *
            E_MINIBUS_EV
            *
            elektrik_fiyat
        )

        yakit = (dizel_yakit + elektrik) * katsayi

        bakim = (
            senaryo["otobus_dizel"] * bakim_otobus_dizel
            +
            senaryo["minibus_dizel"] * bakim_minibus_dizel
            +
            senaryo["otobus_ev"] * bakim_otobus_ev
            +
            senaryo["minibus_ev"] * bakim_minibus_ev
        ) * katsayi

        toplam = yakit + bakim

        if yil == 1:
            toplam += yatirim

        yakit_liste.append(yakit)
        toplam_liste.append(toplam)

    df = pd.DataFrame({
        "Yıl": yillar,
        "Yakıt": yakit_liste,
        "Toplam": toplam_liste
    })

    return df

# =========================================================
# SENARYOLAR
# =========================================================

md = {
    "otobus_dizel": n_otobus,
    "otobus_ev": 0,
    "minibus_dizel": n_minibus,
    "minibus_ev": 0
}

s1 = {
    "otobus_dizel": n_otobus*(2/3),
    "otobus_ev": n_otobus*(1/3),
    "minibus_dizel": n_minibus*(2/3),
    "minibus_ev": n_minibus*(1/3)
}

s2 = {
    "otobus_dizel": n_otobus*(1/3),
    "otobus_ev": n_otobus*(2/3),
    "minibus_dizel": n_minibus*(1/3),
    "minibus_ev": n_minibus*(2/3)
}

s3 = {
    "otobus_dizel": 0,
    "otobus_ev": n_otobus,
    "minibus_dizel": 0,
    "minibus_ev": n_minibus
}

# =========================================================
# HESAPLAMA
# =========================================================

if hesapla:

    em_md = emisyon_hesapla(md)
    em_s1 = emisyon_hesapla(s1)
    em_s2 = emisyon_hesapla(s2)
    em_s3 = emisyon_hesapla(s3)

    df_md = maliyet_hesapla(md)
    df_s1 = maliyet_hesapla(s1)
    df_s2 = maliyet_hesapla(s2)
    df_s3 = maliyet_hesapla(s3)

    # =====================================================
    # METRİKLER
    # =====================================================

    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric(
            "MD CO₂e",
            f"{em_md['CO2e']:.1f} ton"
        )

    with c2:
        st.metric(
            "S1 CO₂e",
            f"{em_s1['CO2e']:.1f} ton"
        )

    with c3:
        st.metric(
            "S2 CO₂e",
            f"{em_s2['CO2e']:.1f} ton"
        )

    with c4:
        st.metric(
            "S3 CO₂e",
            f"{em_s3['CO2e']:.1f} ton"
        )

    st.markdown("---")

    # =====================================================
    # TABLAR
    # =====================================================

    tab1,tab2,tab3 = st.tabs([
        "♻️ Emisyon",
        "💰 Maliyet",
        "📊 Veri"
    ])

    # =====================================================
    # EMİSYON
    # =====================================================

    with tab1:

        fig,ax = plt.subplots(figsize=(10,5))

        labels = ["MD","S1","S2","S3"]

        values = [
            em_md["CO2e"],
            em_s1["CO2e"],
            em_s2["CO2e"],
            em_s3["CO2e"]
        ]

        colors = [
            RENKLER["MD"],
            RENKLER["S1"],
            RENKLER["S2"],
            RENKLER["S3"]
        ]

        bars = ax.bar(
            labels,
            values,
            color=colors
        )

        ax.set_title("Senaryo Bazlı CO₂e Emisyonu")
        ax.set_ylabel("Ton CO₂e")

        for bar,val in zip(bars,values):

            ax.text(
                bar.get_x()+bar.get_width()/2,
                val,
                f"{val:.1f}",
                ha='center',
                va='bottom'
            )

        st.pyplot(fig)

    # =====================================================
    # MALİYET
    # =====================================================

    with tab2:

        fig2,ax2 = plt.subplots(figsize=(11,5))

        ax2.plot(
            df_md["Yıl"],
            df_md["Toplam"].cumsum()/1e6,
            label="MD",
            linewidth=3
        )

        ax2.plot(
            df_s1["Yıl"],
            df_s1["Toplam"].cumsum()/1e6,
            label="S1",
            linewidth=3
        )

        ax2.plot(
            df_s2["Yıl"],
            df_s2["Toplam"].cumsum()/1e6,
            label="S2",
            linewidth=3
        )

        ax2.plot(
            df_s3["Yıl"],
            df_s3["Toplam"].cumsum()/1e6,
            label="S3",
            linewidth=3
        )

        ax2.set_title("Kümülatif Toplam Maliyet")
        ax2.set_xlabel("Yıl")
        ax2.set_ylabel("Milyon TL")

        ax2.grid(True)
        ax2.legend()

        st.pyplot(fig2)

    # =====================================================
    # TABLO
    # =====================================================

    with tab3:

        secim = st.selectbox(
            "Senaryo Seç",
            ["MD","S1","S2","S3"]
        )

        tablo = {
            "MD":df_md,
            "S1":df_s1,
            "S2":df_s2,
            "S3":df_s3
        }

        st.dataframe(
            tablo[secim],
            use_container_width=True
        )
