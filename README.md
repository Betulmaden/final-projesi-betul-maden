# Telco Müşteri Kayıp (Churn) Tahmin Projesi 📊

**Projenin Adı:** Telco Müşteri Kayıp (Churn) Tahmini ve Karar Destek Sistemi

**Problem Tanımı:** Şirketler, müşterilerinin ne zaman abonelik iptal edeceğini (churn) önceden bilemiyor. Bu yüzden müşteriyi elde tutmak için kampanya veya indirim yapma fırsatını çoğu zaman kaçırıyorlar. Bu projenin amacı, bu kayıpları olmadan önce tahmin edebilmek.

**Hedef Kullanıcı:** Şirketlerin müşteri ilişkileri (CRM) departmanları ve veri analistleri.

**Çözümün Kısa Açıklaması:** Geçmiş müşteri verilerini kullanarak, makine öğrenmesi ile bir müşterinin ayrılma riskini tahmin eden bir web arayüzü geliştirdim. Sadece oran vermekle kalmadım, modelin bu kararı neden verdiğini de (SHAP kütüphanesi kullanarak) görselleştirdim. Böylece arayüz, "Bu müşteri ayrılacak çünkü aylık faturası çok yüksek" gibi mantıklı açıklamalar yapabiliyor.

**Kullanılan Teknolojiler:**
* **Veri Hazırlığı:** Python, Pandas, NumPy
* **Makine Öğrenmesi:** Scikit-Learn (Random Forest, Logistic Regression)
* **Açıklanabilirlik:** SHAP, Matplotlib
* **Arayüz:** Streamlit

**Sistem Mimarisi ve İş Akışı:**
1. Kaggle'dan aldığım orijinal veri setini `model_egitim.py` dosyasında temizledim.
2. Modelleri eğittim ve en yüksek doğruluk skorunu alanı ileride kullanmak üzere `.pkl` olarak kaydettim.
3. `app.py` ile bu modeli bir Streamlit web arayüzüne bağladım. Arayüzden girilen müşteri özellikleri anlık olarak modele gidiyor ve bize sonuç döndürüyor.

**Kurulum Adımları:**
Projeyi kendi bilgisayarınızda çalıştırmak için terminale şu komutu yazarak kütüphaneleri kurmanız yeterli:
`pip install pandas numpy scikit-learn shap streamlit matplotlib`

**Kullanım Biçimi:**
Terminalden projenin olduğu klasöre girip `streamlit run app.py` yazın. Tarayıcıda açılan sayfada, sol taraftaki menüden müşteri özelliklerini (fatura, cinsiyet, süre vb.) girip "Tahmin Et" butonuna tıklayabilirsiniz.

**Örnek Ekran Görüntüleri:**
*(Buraya arayüzün ekran görüntüsünü yükleyeceğim)*

**Test Sonuçları:**
Veriyi %80 eğitim, %20 test olarak ayırdım. Logistic Regression modeli %78 doğruluk (accuracy) verirken, Random Forest modeli %79 verdi. Bu yüzden Random Forest ile ilerledim. Arayüzü uç değerlerle (hiç faturası olmayan veya çok eski müşterilerle) denediğimde de sistem hata vermeden çalıştı.

**Bilinen Sınırlılıklar:**
Sistem şu an statik bir csv dosyası üzerinden eğitildi, canlı bir şirket veritabanına bağlı değil. Bir de SHAP grafikleri hesaplamaları biraz ağır olduğu için web sayfasında sonucu verirken birkaç saniye bekletebiliyor.

**Gelecekte Yapılabilecek Geliştirmeler:**
Kullanıcıların tek tek veri girmesi yerine, sisteme toplu bir excel yükleyip aynı anda yüzlerce müşterinin risk durumunu tek seferde analiz eden bir özellik eklenebilir. 

**Yapay Zekâ Araçlarının Hangi Aşamalarda Kullanıldığı:**
Projeyi geliştirirken kodlamada takıldığım yerleri aşmak ve hata ayıklamak (debugging) için Google Gemini kullandım. Özellikle Streamlit arayüzünü ayağa kaldırırken ve SHAP grafiklerinde yaşadığım boyut uyuşmazlığı hatalarında Gemini bana yol gösterdi. Kodların mantığını kendim kavrayıp yerel ortamımda (Mac terminalimde) test ederek ilerledim.

---
🎥 **Proje Tanıtım Videosu:** 
