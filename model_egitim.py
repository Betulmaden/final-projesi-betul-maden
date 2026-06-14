import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
import pickle

print("1. Veri Yükleme ve Temizleme Aşaması...")
# Excel'in virgül hatasına takılmadan orijinal CSV dosyasını okuyoruz
df = pd.read_csv('customer_churn.csv')

# Orijinal verideki boşlukları düzeltip hatalı satırları siliyoruz
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Kategorik verileri yapay zekanın anlayacağı 0 ve 1'lere dönüştürüyoruz
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['Partner'] = df['Partner'].map({'No': 0, 'Yes': 1})
df['Dependents'] = df['Dependents'].map({'No': 0, 'Yes': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Modelin kullanacağı sayısal özellikleri (feature) belirliyoruz
X = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MonthlyCharges']]
y = df['Churn']

print("2. Veri Setini Train-Test Olarak Ayırma...")
# Veriyi %80 eğitim, %20 test olacak şekilde ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("3. Modellerin Eğitilmesi ve Karşılaştırılması...")
# Model 1: Random Forest (Overfitting'i engellemek için max_depth=5 sınırı koyduk)
rf_model = RandomForestClassifier(random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# Model 2: Logistic Regression (max_iter=1000 ile yakınsama sorunu çözüldü)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print(f"-> Random Forest Doğruluk Skoru: {rf_acc:.2f}")
print(f"-> Logistic Regression Doğruluk Skoru: {lr_acc:.2f}")

# En iyi skoru veren modeli bulup .pkl dosyası olarak kaydediyoruz
en_iyi_model = rf_model if rf_acc >= lr_acc else lr_model
with open('en_iyi_model.pkl', 'wb') as f:
    pickle.dump(en_iyi_model, f)
print("En iyi model 'en_iyi_model.pkl' olarak başarıyla kaydedildi.")

print("4. SHAP Açıklayıcı Yapay Zeka Bileşeni...")
# En iyi modele uygun SHAP açıklayıcısını kurup kaydediyoruz
if en_iyi_model == rf_model:
    explainer = shap.TreeExplainer(rf_model)
else:
    explainer = shap.LinearExplainer(lr_model, X_train)

with open('shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)
print("SHAP altyapısı başarıyla oluşturuldu ve kaydedildi.")
print("Eğitim süreci başarıyla tamamlandı!")
