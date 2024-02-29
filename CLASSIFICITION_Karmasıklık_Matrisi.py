# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:58:20 2024

@author: ahmet
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Örnek veri setini yükleme veya kullanılan veri setini yükleme
df = pd.read_csv("yeni_mushrooms.csv")
# =============================================================================
# 
# df['prediction_result'] = df['prediction'].apply(lambda x: 1 if x == 0 else 0)
# 
# # Dönüştürülmüş veriyi görüntüleme
# print(df.head())
# =============================================================================

X = df.drop('class', axis=1)  # 'class' sütunu haricindeki özellikler
y = df['prediction']  # 'class' sütunu (etiket)

# Veriyi eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

# Model oluşturma ve eğitme (örneğin, RandomForest kullanıldı)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test setinde tahminler yapma
predictions = model.predict(X_test)

# Modelin başarısını değerlendirme
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# Tüm veri seti üzerinde tahminler yapma
all_predictions = model.predict(X)

# Tahminleri yeni bir sütun olarak ekleyin
df['prediction'] = all_predictions

# Dönüştürülmüş veriyi görüntüleme
print(df.head())



# Örnek veri ve etiketler


# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

# Bir sınıflandırma modeli oluşturma ve eğitme
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahminler yapma
y_pred = model.predict(X_test)

# Karmaşıklık matrisini hesaplama
cm = confusion_matrix(y_test, y_pred)

# Diğer performans metrikleri
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Değer')
plt.show()

# Dönüştürülmüş veriyi görüntüleme
#print(df.head())

df.to_csv('yeni_mushrooms.csv', index=False)