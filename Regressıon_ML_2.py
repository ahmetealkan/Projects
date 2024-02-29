# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:06:40 2024

@author: ahmet
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Veri setini yükleme
df = pd.read_csv('yeni_student-por.csv')

# Özellikler (X) ve etiket (y) ayırma
X = df.drop('Basari_Durumu', axis=1)
y = df['Basari_Durumu']

# Belirtilen sayıda tekrarlamayı gerçekleştirin
num_repeats = 100
accuracy_values = []

for _ in range(num_repeats):
    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

    # Karar Ağacı modelini oluşturma ve eğitme
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    # Model performansını değerlendirme ve doğruluk değerini listeye ekleme
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)

# Doğruluk değerlerinin istatistiksel özelliklerini hesaplama
accuracy_mean = np.mean(accuracy_values)
accuracy_std = np.std(accuracy_values)
highest_accuracy = np.max(accuracy_values)
lowest_accuracy = np.min(accuracy_values)

# Sonuçları yazdırma
print(f'Doğruluk (Ortalama): {accuracy_mean:.2f}')
print(f'Doğruluk (Standart Sapma): {accuracy_std:.2f}')
print(f'Doğruluk (En Yüksek): {highest_accuracy:.2f}')
print(f'Doğruluk (En Düşük): {lowest_accuracy:.2f}')



