import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# 1. Sayfa Ayarları ve Başlık
st.set_page_config(page_title="Müşteri Kayıp Tahmini", layout="wide")
st.title("Telco Müşteri Kayıp (Churn) Tahmin Sistemi")
st.markdown("Bu karar destek arayüzü, makine öğrenmesi modeli kullanarak müşterilerin şirketi terk etme riskini analiz eder.")

# 2. Arka Planda Modelleri Yükleme
with open('en_iyi_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('shap_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

# 3. Sol Menü (Kullanıcı Veri Girişi)
st.sidebar.header("Müşteri Bilgilerini Giriniz")
gender = st.sidebar.selectbox("Cinsiyet", ["Kadın", "Erkek"])
senior_citizen = st.sidebar.selectbox("Yaşlı Müşteri mi?", ["Hayır", "Evet"])
partner = st.sidebar.selectbox("Partneri/Eşi Var mı?", ["Hayır", "Evet"])
dependents = st.sidebar.selectbox("Bakmakla Yükümlü Olduğu Biri Var mı?", ["Hayır", "Evet"])
tenure = st.sidebar.slider("Müşterilik Süresi (Ay)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Aylık Fatura Tutarı ($)", min_value=0.0, max_value=200.0, value=50.0)

# Kullanıcının girdiği verileri modelin anlayacağı sayılara dönüştürme
input_data = pd.DataFrame({
    'gender': [0 if gender == "Kadın" else 1],
    'SeniorCitizen': [0 if senior_citizen == "Hayır" else 1],
    'Partner': [0 if partner == "Hayır" else 1],
    'Dependents': [0 if dependents == "Hayır" else 1],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges]
})

# 4. Tahmin Butonu ve Sonuç Ekranı
if st.sidebar.button("Tahmin Et"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.subheader("Tahmin Sonucu")
    if prediction == 1:
        st.error(f"⚠️ Bu müşterinin şirketi TERK ETME riski yüksek! (Olasılık: %{prediction_proba[1]*100:.1f})")
    else:
        st.success(f"✅ Bu müşteri şirkette KALACAK gibi görünüyor. (Kalma olasılığı: %{prediction_proba[0]*100:.1f})")
        
    # SHAP ile Açıklanabilirlik Grafiği
    st.subheader("Model Bu Kararı Neden Verdi? (SHAP Açıklaması)")
    st.info("Aşağıdaki grafik, müşterinin özelliklerinin tahmini nasıl etkilediğini gösterir. (Kırmızı: Ayrılmaya iter, Mavi: Kalmaya iter)")
    
    # SHAP Değerlerini Hesaplama
    shap_values = explainer(input_data)
    
    # Random Forest modeli çok boyutlu dizi döndürdüğü için ayrılma (Class 1) boyutunu seçiyoruz
    if len(shap_values.shape) == 3:
        explanation = shap_values[:, :, 1][0] 
    else:
        explanation = shap_values[0]
        
    # Grafiği güvenli bir şekilde çizdirme
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig)
