import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

# Örnek veri setini yükleme veya kullanılan veri setini yükleme
df = pd.read_csv('Toddler_Autism_dataset_July_2018_new.csv')

# X ve y'yi uygun şekilde tanımlayın
X = df.drop('Class/ASD Traits ', axis=1)  # "class" sütunu hedef değişken, geri kalanlar özellikler
y = df['Class/ASD Traits ']

# Kullanılacak küme sayısı
n_clusters = 2

# Tekrar sayısı
num_repeats = 100

# Performans metriklerini depolamak için boş listeler
silhouette_scores = []
best_score = -1
worst_score = 1

# Belirtilen sayıda tekrarlamayı gerçekleştirin
for _ in range(num_repeats):
    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test = train_test_split(X, test_size=0.4, random_state=None)

    # Agglomerative Clustering modelini oluşturma ve eğitme
    model = AgglomerativeClustering(n_clusters=n_clusters)
    model.fit(X_train)

    # Küme etiketlerini alarak Silhouette skorunu hesaplama
    silhouette_avg = silhouette_score(X_train, model.labels_)
    silhouette_scores.append(silhouette_avg)

    # En iyi ve en kötü skoru güncelleme
    best_score = max(best_score, silhouette_avg)
    worst_score = min(worst_score, silhouette_avg)

# Ortalama Silhouette skoru hesaplama
average_score = np.mean(silhouette_scores)

# Standart sapma hesaplama
std_dev_score = np.std(silhouette_scores)

# Sonuçları yazdırma
print(f"Ortalama Silhouette Score: {average_score}")
print(f"Standart Sapma: {std_dev_score}")
print(f"En İyi Silhouette Score: {best_score}")
print(f"En Kötü Silhouette Score: {worst_score}")
