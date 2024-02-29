# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:38:47 2024

@author: ahmet
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import random

# Örnek veri setini yükleme veya kullanılan veri setini yükleme
df = pd.read_csv("yeni_mushrooms.csv")
# X ve y'yi uygun şekilde tanımlayın


# bu kısmı halletmemiz lazım son aldığım hata sonuc diye bir başlığın olmaması 
X = df.drop('class', axis=1)  # "class" sütunu hedef değişken, geri kalanlar özellikler
y = df['class']

random_indices = np.random.permutation(df.index)
df = df.reindex(random_indices)


# Tekrar sayısı
num_repeats = 100

# Performans metriklerini depolamak için boş listeler
accuracies = []
best_accuracy = 0
worst_accuracy = 1

# Belirtilen sayıda tekrarlamayı gerçekleştirin
for _ in range(num_repeats):
    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=None)

    # Karar ağacı modelini eğitme
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Test seti üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    # Accuracy hesaplama
    accuracy = accuracy_score(y_test, y_pred)

    # En iyi ve en kötü başarıyı güncelleme
    best_accuracy = max(best_accuracy, accuracy)
    worst_accuracy = min(worst_accuracy, accuracy)

    # Accuracy'yi listeye ekleme
    accuracies.append(accuracy)

# Ortalama accuracy hesaplama
average_accuracy = np.mean(accuracies)

# Standart sapma hesaplama
std_dev_accuracy = np.std(accuracies)

# Sonuçları yazdırma
print(f"Ortalama Accuracy: {average_accuracy}")
print(f"Standart Sapma: {std_dev_accuracy}")
print(f"En İyi Accuracy: {best_accuracy}")
print(f"En Kötü Accuracy: {worst_accuracy}")
