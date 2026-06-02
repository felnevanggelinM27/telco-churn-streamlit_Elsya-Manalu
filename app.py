import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set konfigurasi halaman agar responsif dan modern
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# 1. Load model dan encoders langsung dari root direktori
@st.cache_resource
def load_models():
    model = joblib.load("churn_model.pkl")
    encoders = joblib.load("encoders.pkl")
    return model, encoders

model, encoders = load_models()

# Load dataset utama langsung dari root direktori
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df

try:
    df_clean = load_data()
except FileNotFoundError:
    st.error("File 'WA_Fn-UseC_-Telco-Customer-Churn.csv' tidak ditemukan. Harap periksa repositori GitHub Anda.")
    st.stop()


# ==========================================
# SIDEBAR - INPUT FORM UNTUK PREDIKSI SINGLE
# ==========================================
st.sidebar.header("🔌 Input Fitur Pelanggan")
st.sidebar.write("Masukkan detail profil untuk mengecek risiko churn.")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure (Durasi Kontrak/Bulan)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])

# Data default pelengkap agar sesuai dengan training format
input_data = pd.DataFrame({
    "gender": [gender], "SeniorCitizen": [0], "Partner": ["No"], "Dependents": ["No"],
    "tenure": [tenure], "PhoneService": ["Yes"], "MultipleLines": ["No"],
    "InternetService": [internet], "OnlineSecurity": ["No"], "OnlineBackup": ["No"],
    "DeviceProtection": ["No"], "TechSupport": ["No"], "StreamingTV": ["No"],
    "StreamingMovies": ["No"], "Contract": [contract], "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"], "MonthlyCharges": [monthly],
    "TotalCharges": [float(monthly * tenure)]
})

# Jalankan encoding data input
for col in input_data.columns:
    if col in encoders:
        try:
            input_data[col] = encoders[col].transform(input_data[col])
        except ValueError:
            input_data[col] = 0


# ==========================================
# MAIN PAGE - LAYOUT UTAMA DASHBOARD
# ==========================================
st.title("📊 Customer Churn Intelligence Dashboard")
st.markdown("Dasbor berbasis Machine Learning untuk mendeteksi risiko retensi pelanggan dan visualisasi tren bisnis.")

# Buat Tab agar dashboard rapi (Model Prediction vs Business Analytics)
tab1, tab2 = st.tabs(["🔮 Prediksi Risiko Pelanggan", "📈 Analitik & Evaluasi Model"])

# ------------------------------------------
# TAB 1: INDIVIDUAL PREDICTION & GAUGE CHART
# ------------------------------------------
with tab1:
    st.subheader("Model Inference System")
    
    # Tombol Prediksi
    if st.sidebar.button("⚙️ Jalankan Prediksi Churn", use_container_width=True):
        # Jalankan kalkulasi prediksi & probabilitas
        prediction = model.predict(input_data)[0]
        prob_churn = model.predict_proba(input_data)[0][1] # Ambil probabilitas kelas 1 (Churn)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### Status Kelayakan")
            if prediction == 1:
                st.error(f"⚠️ **CUSTOMER BERPOTENSI CHURN**")
                st.markdown(f"Pelanggan ini memiliki indikasi kuat untuk berhenti berlangganan. Diperlukan tindakan retensi secepatnya.")
            else:
                st.success(f"✅ **CUSTOMER CENDERUNG SETIA (LOYAL)**")
                st.markdown(f"Pelanggan berada dalam zona aman. Pertahankan kualitas pelayanan saat ini.")
                
            # Informasi Pendukung Singkat
            st.metric(label="Estimasi Total Pendapatan Kehilangan/Kontrak", value=f"${monthly * tenure:,.2f}")
            
        with col2:
            st.markdown("### Skor Tingkat Risiko (Risk Score)")
            # Gauge Chart menggunakan Plotly Graph Objects
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob_churn * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probabilitas Churn (%)", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#3A3B3C"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#2ecc71'}, # Hijau (Rendah)
                        {'range': [40, 70], 'color': '#f1c40f'}, # Kuning (Sedang)
                        {'range': [70, 100], 'color': '#e74c3c'} # Merah (Tinggi)
                    ],
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.info("💡 Silakan klik tombol **'Jalankan Prediksi Churn'** pada sidebar kiri untuk mulai menguji data pelanggan.")


# ------------------------------------------
# TAB 2: ANALYTICS, KPI CARDS & EVALUATION
# ------------------------------------------
with tab2:
    st.subheader("Business Key Performance Indicators (KPI)")
    
    # Kalkulasi metrik sederhana dari dataset asli untuk dipasang di KPI Cards
    total_customers = len(df_clean)
    churn_rate = (df_clean['Churn'].value_counts(normalize=True).get('Yes', 0)) * 100
    avg_monthly = df_clean['MonthlyCharges'].mean()
    
    # Tampilkan KPI Cards secara horizontal
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Total Database Pelanggan", value=f"{total_customers:,}")
    kpi2.metric(label="Rasio Churn Historis", value=f"{churn_rate:.2%}")
    kpi3.metric(label="Rerata Tagihan Bulanan", value=f"${avg_monthly:.2f}")
    kpi4.metric(label="Akurasi Model ML (RF)", value="79.31%", delta="Production Ready")
    
    st.write("---")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("### Komposisi Rasio Churn Keseluruhan")
        # Pie Chart Proporsi Churn
        fig_pie = px.pie(
            df_clean, 
            names='Churn', 
            hole=0.4,
            color='Churn',
            color_discrete_map={'No':'#2ecc71', 'Yes':'#e74c3c'}
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.markdown("### Analisis Churn Berdasarkan Jenis Kontrak")
        # Histogram Churn vs Kontrak
        fig_hist = px.histogram(
            df_clean,
            x="Contract",
            color="Churn",
            barmode="group",
            color_discrete_map={'No':'#2ecc71', 'Yes':'#e74c3c'}
        )
        fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=320)
        st.plotly_chart(fig_hist, use_container_width=True)
