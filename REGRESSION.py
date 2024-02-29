# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:58:36 2024

@author: ahmet
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv('student-por.csv')



df['Basari_Durumu']=df['G1']+df['G2']+df['G3']

df['Basari_Durumu'] = df['Basari_Durumu'].astype(int)





df['school'] = df['school'].map({'GP':1,'MS': 0})
df['sex'] = df['sex'].map({'F':1,'M': 0})
df['address'] = df['address'].map({'U':1,'R': 0})
df['famsize'] = df['famsize'].map({'GT3':1,'LE3': 0})
df['Pstatus'] = df['Pstatus'].map({'T':1,'A': 0})
df['guardian'] = df['guardian'].map({'mother':0,'father':1,'other': 2})
df['reason'] = df['reason'].map({'other ': 0,'home': 1,'reputation': 2,'other': 3})

df['Mjob'] = df['Mjob'].map({'other ': 4,'services ':1,'at_home': 2,'teacher ': 3,'health':3})
df['Fjob'] = df['Fjob'].map({'other ': 4,'services ':1,'at_home': 2,'teacher ': 3,'health':3})



df['Basari_Durumu'] = np.where(df['Basari_Durumu'] < 28, 0, 1)
df['Basari_Durumu'] = df['Basari_Durumu'].astype(int)

dummies=['schoolsup','famsup','paid','activities','nursery',
         'higher','internet','romantic',]   


for c in dummies:
    dummies_df = pd.get_dummies(df[c], prefix=c)
    df = pd.concat([df, dummies_df], axis=1)
    df.drop(c, axis=1, inplace=True)



# =============================================================================
# selected_columns = ['school','sex','age','address','famsize',
#                     'Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian',
#                     'traveltime','studytime','failures','famrel','freetime','goout',
#                    'Walc','health','absences','G1','G2','G3','schoolsup_no',
#                     'schoolsup_yes','famsup_no','famsup_yes','paid_no','paid_yes','activities_no',
#                     'activities_yes','nursery_no','nursery_yes','higher_no','higher_yes','internet_no',
#                     'internet_yes','romantic_no','romantic_yes','Basari_Durumu']
# =============================================================================

# school - adress tamamdır, ,faiuleres-G1,G2,G3,higher_no,higher_no ekle korelasyona bak
#higher_no ile G2 ve G3 olsun 

# Bu sütunun değerleri 1-5 arasında olmalıdır
df['Dalc'] = df['Dalc'].astype(int)

# 'Alkol_Sorunu' adında yeni bir sütun oluşturun
# Koşullara göre dönüşümü gerçekleştirin
#Dalc- Walc
df['Alkol_Sorunu'] = df['Dalc'].apply(lambda x: 0 if x in [1, 2] else 1)

df['Walc'] = df['Walc'].astype(int)

# 'Alkol_Sorunu' adında yeni bir sütun oluşturun
# Koşullara göre dönüşümü gerçekleştirin
df['Alkol_Sorunu2'] = df['Walc'].apply(lambda x: 0 if x in [1, 2] else 1)

# Örneğin, 'Alkol_Sorunu' sütunu 1'den büyük VE 'Alkol_Sorunu2' sütunu 0'dan küçükse 'Alkol_Sorunu3' 1 olacak, aksi takdirde 0 olacak.
df['Alkol_Sorunu3'] = (df['Alkol_Sorunu'] == 1) & (df['Alkol_Sorunu2'] < 0)

#df['Alkol_Sorunu3'] = df['Alkol_Sorunu3'].map({'True':1,'False': 0})

ortalama_basari_durumu = df['failures'].unique()
sayi_dagilimi = df['failures'].value_counts()


# G1+G2,G1+G3,G2+G3  --> bunları bir sınır belirle onun altındakilere 0 diğerlerine 1   ekle korelasyona bak

df['G1+G2'] = df['G1'] + df['G2']

# Yeni bir sütun oluşturma ve koşulu uygulama
df['Sonuc_G12'] = (df['G1+G2'] / 2 > 9).astype(int)

df['G1+G3'] = df['G1'] + df['G3']

# Yeni bir sütun oluşturma ve koşulu uygulama
df['Sonuc_G13'] = (df['G1+G3'] / 2 > 9).astype(int)

df['G2+G3'] = df['G2'] + df['G3']

# Yeni bir sütun oluşturma ve koşulu uygulama
df['Sonuc_G23'] = (df['G2+G3'] / 2 > 9).astype(int)


del df['G1+G2']
del df['G2+G3']
del df['G1+G3']
# Sonuçları gösterme
print(df)


# Elde edilen ortalamayı yazdır



# =============================================================================
# df['school+adress'] = df['school'] + df['adress']
# 
# # Koşula göre değerleri güncelleyin
# df['school+adress'] = df['school+adress'].apply(lambda x: 1 if x > 0 else 0)
# 
# =============================================================================
# Sonuçları yazdırma
print(df.head())



del df['Fjob']

selected_columns = ['school','address','reason',
                    'failures','Dalc','freetime',
                   'Walc','absences','G1','G2','G3','higher_no',
                   'higher_yes','Sonuc_G12','Sonuc_G13','Sonuc_G23','Basari_Durumu']


df_selected = df[selected_columns]





# Kategorik değişkenleri sayısala çevirme
df_numeric = df_selected.apply(lambda x: pd.factorize(x)[0])
 
# Korelasyon matrisini oluşturma
correlation_matrix = df_numeric.corr()

# Heatmap oluşturma
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Özelliklerin 'class' ile Korelasyonu Heatmap")
plt.show()


korelasyonlar = df.corr()['Basari_Durumu']

# Eşik değeri belirle
esik_degeri = 0.1

# Eşik değerinden küçük korelasyona sahip sütunları seç
dusuk_korelasyonlu_sutunlar = korelasyonlar[abs(korelasyonlar) < esik_degeri].index

# DataFrame'den düşük korelasyonlu sütunları kaldır
df = df.drop(dusuk_korelasyonlu_sutunlar, axis=1)

# Sonuçları yazdırma
print("Düşük korelasyonlu sütunlar kaldırıldı:")
print(df.head())

print(f"eşsiz değerler: {ortalama_basari_durumu}")
print(f"Sayı dağılımı: {sayi_dagilimi}")

nan_rows = df[df.isnull().any(axis=1)]

# NaN değerlere sahip sütunları kontrol et
nan_columns = df.columns[df.isnull().any()]

# Sonuçları göster
print("NaN değerlere sahip örnekler:")
print(nan_rows)

print("\nNaN değerlere sahip sütunlar:")
print(nan_columns)



del df['reason']

df.to_csv('yeni_student-por.csv', index=False)