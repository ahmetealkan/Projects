# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:30:10 2024

@author: ahmet
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

df = pd.read_csv("Toddler Autism dataset July 2018.csv")

correlation_matrix = df.corr()

# Korelasyon matrisini ısı haritasıyla görselleştirme



def aykiri_deger_sil( df: pd.DataFrame, col: str, limit: float = 0.01 ) -> pd.DataFrame:
    Q3 = df[col].quantile(0.75)#3 cu ceyrek 
    Q1 = df[col].quantile(0.25)#1 ceyrek
    IQR = Q3 - Q1
    alt_limit= Q1 - 1.5 * IQR
    ust_limit = Q3 + 1.5 * IQR
    ust_aykiri_sayisi= len(df[ df[col] > ust_limit ]) / len(df)#ust_limiti asan aykiri sayisi
    alt_aykiri_sayisi= len(df[ df[col] < alt_limit ]) / len(df)#alt limitin altinda kalan aykiri sayisi

    if ust_aykiri_sayisi < limit:
        df = df[df[col] <= ust_limit]# yukaridaki outlier siliyor
    if alt_aykiri_sayisi < limit:
        df = df[df[col] >= alt_limit]#asagidaki outlier siliyor
    return df

df['Class/ASD Traits '] = df['Class/ASD Traits '].map({'Yes': 1, 'No': 0})



unique_values = df['Ethnicity'].unique()
print("Eşsiz Değerler:", unique_values)

veri_listesi = ['family member','Health Care Professional','Health care professional', 'Self','Others']

eşlemeler = {
    'family member': '1',
    'Health Care Professional': '2',
    'Health care professional': '2',  # Büyük küçük harf karışıklığına karşı iki giriş
    'Self': '3',
    'Others': '4'
}

df['Who completed the test'] = df['Who completed the test'].map({'family member': 1,
'Health Care Professional': 2,
'Health care professional': 2,  # Büyük küçük harf karışıklığına karşı iki giriş
'Self': 3,
'Others': 4})

# =============================================================================
# [   Native Indian' 'Others' 'mixed' 'Pacifica']
# White European ; 1; south asian ,asian,'middle eastern': 2;'black' :3,'black' :4;'Native Indian','Others','mixed','Pacifica':5;
# =============================================================================

dummies=['Sex','Jaundice','Family_mem_with_ASD',]

value_counts = df['Ethnicity'].value_counts()
print(value_counts)

df['Ethnicity']=df['Ethnicity'].map({'White European' : 1, 'south asian': 2 ,'asian': 2,'middle eastern': 2,'black' :3,
                                     'black' :4,'Native Indian':5,'Others':5,'mixed':5,'Pacifica':5,'Hispanic':5,'Latino':5})

## yaşları da 12 24 arası 0 olsun 24 36 arası 1 olsun onun algosunu yazcaksın  

df['Age_Mons'] = df['Age_Mons'].apply(lambda x: 0 if 12 <= x < 24 else (1 if 24 <= x < 36 else 2))

# İlk beş satırı gösterme
print(df.head())


for c in dummies:
    dummies_df = pd.get_dummies(df[c], prefix=c)
    df = pd.concat([df, dummies_df], axis=1)
    df.drop(c, axis=1, inplace=True)
    
    
df['A6+A5'] = df['A6'] + df['A5']

# Koşula göre değerleri güncelleyin
df['A6+A5'] = df['A6+A5'].apply(lambda x: 1 if x > 0 else 0)   

df['A9+A1'] = df['A9'] + df['A1']

# Koşula göre değerleri güncelleyin
df['A9+A1'] = df['A9+A1'].apply(lambda x: 1 if x > 0 else 0)  

df['A9+A5'] = df['A9'] + df['A5']

# Koşula göre değerleri güncelleyin
df['A9+A5'] = df['A9+A5'].apply(lambda x: 1 if x > 0 else 0)  

df['A3+A4'] = df['A3'] + df['A4']

# Koşula göre değerleri güncelleyin
df['A3+A4'] = df['A3+A4'].apply(lambda x: 1 if x > 0 else 0)  

df['A6+A1'] = df['A6'] + df['A1']

# Koşula göre değerleri güncelleyin
df['A6+A1'] = df['A6+A1'].apply(lambda x: 1 if x > 0 else 0)  

df['A1+A2'] = df['A1'] + df['A2']

# Koşula göre değerleri güncelleyin
df['A1+A2'] = df['A1+A2'].apply(lambda x: 1 if x > 0 else 0)  

df['A9+A6'] = df['A9'] + df['A6']

# Koşula göre değerleri güncelleyin
df['A9+A6'] = df['A9+A6'].apply(lambda x: 1 if x > 0 else 0)

# 7 - 6, 5-4,5-3,4-3
df['A7+A6'] = df['A7'] + df['A6']

# Koşula göre değerleri güncelleyin
df['A7+A6'] = df['A7+A6'].apply(lambda x: 1 if x > 0 else 0)

df['A5+A4'] = df['A5'] + df['A4']

# Koşula göre değerleri güncelleyin
df['A5+A4'] = df['A5+A4'].apply(lambda x: 1 if x > 0 else 0)

df['A5+A3'] = df['A5'] + df['A3']

# Koşula göre değerleri güncelleyin
df['A5+A3'] = df['A5+A3'].apply(lambda x: 1 if x > 0 else 0)





silinecekler = ['Age_Mons','Qchat-10-Score','Ethnicity',
'Who completed the test','Sex_f','Sex_m','Jaundice_no','Jaundice_yes',
'Family_mem_with_ASD_no','Family_mem_with_ASD_yes']

for c in silinecekler :
  del[c]
selected_columns = ['A1','A2','A3','A4',
                    'A5','A6','A7','A8','A9','A10',
                    'A9+A1','A6+A5','A9+A5','A5+A3','A5+A4','A7+A6',
                    'A3+A4','A6+A1','A1+A2','A9+A6','Class/ASD Traits ']
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




nan_rows = df[df.isnull().any(axis=1)]

# NaN değerlere sahip sütunları kontrol et
nan_columns = df.columns[df.isnull().any()]

# Sonuçları göster
print("NaN değerlere sahip örnekler:")
print(nan_rows)

print("\nNaN değerlere sahip sütunlar:")
print(nan_columns)

df.to_csv('Toddler_Autism_dataset_July_2018_new.csv', index=False)