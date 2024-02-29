import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import numpy as np

# CSV dosyasını oku (veri setinizi uygun şekilde yükleyin)
df = pd.read_csv('yeni_student-por.csv')

# Bağımsız değişkenleri (X) ve bağımlı değişkeni (y) belirle
X = df.drop('Basari_Durumu', axis=1)  
y = df['Basari_Durumu']

# Number of iterations
num_iterations = 100

# Lists to store results
accuracy_scores = []

for i in range(num_iterations):
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=i)
    
    # RandomForestClassifier modelini oluştur
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=i)
    
    # Modeli eğit
    rf_classifier.fit(X_train, y_train)
    
    # Doğruluk değerini hesapla ve listeye ekle
    accuracy = rf_classifier.score(X_test, y_test)
    accuracy_scores.append(accuracy)

# Calculate metrics
accuracy_mean = np.mean(accuracy_scores)
accuracy_std = np.std(accuracy_scores)
highest_accuracy = np.max(accuracy_scores)
lowest_accuracy = np.min(accuracy_scores)

print(f'Doğruluk (Ortalama): {accuracy_mean}')
print(f'Doğruluk (Standart Sapma): {accuracy_std}')
print(f'Doğruluk (En Yüksek): {highest_accuracy}')
print(f'Doğruluk (En Düşük): {lowest_accuracy}')
