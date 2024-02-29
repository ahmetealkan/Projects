# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:58:20 2024

@author: ahmet
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("mushrooms.csv")

#df.head()

#df.describe()
dummies = ['bruises','gill-attachment',
           'gill-spacing','gill-size','stalk-shape',]
          
for c in dummies:
    dummies_df = pd.get_dummies(df[c], prefix=c)
    df = pd.concat([df, dummies_df], axis=1)
    df.drop(c, axis=1, inplace=True)

df['veil-type_partial'] = (df['veil-type'] == 'p').astype(int)
df['veil-color_w'] = (df['veil-color'] == 'w').astype(int)
# =============================================================================
#value_counts = df['cap-shape'].value_counts()
#print(value_counts)
# 
# =============================================================================

cap_shape_mapping = {'x': 1, 'f': 2, 'k': 3, 'b':3, 's': 3, 'c': 3}
df['cap-shape'] = df['cap-shape'].map(cap_shape_mapping)


# Birleştirme işlemi
df['combined_colors'] = df['cap-color'] + df['stalk-color-below-ring'] + df['gill-color']

   



#summary = df.describe(include='all')
#print(summary)




df['combined-stalk-surface'] = df['stalk-surface-above-ring'] + '-' + df['stalk-surface-below-ring']

df['combined-stalk-surface-numeric'] = (df['combined-stalk-surface'] == 's-s').astype(int)
# s-s 4100 tane olduğu için bunu 1 yaptık diğerlerini 0 yaptık 
def renk_kombinasyonunu_kontrol_et(row):
    brown_buff = row['cap-color'] in ['n', 'b']
    cinnamon_orange_red = row['stalk-color-below-ring'] in ['c', 'o', 'e']
    green_orange_purple = row['gill-color'] in ['r', 'o', 'u']
    white_yellow = row['stalk-color-below-ring'] in ['w', 'y']

    if brown_buff or cinnamon_orange_red or green_orange_purple or white_yellow:
        return 1
    else:
        return 0

value_counts= df["ring-type"].value_counts()
print(value_counts)
# ring- type ====================================================================
# p    3968
# e    2776
# l    1296
# f      48
# n      36
# =============================================================================
# ring number ================================================================
# o    7488
# t     600
# n      36
# =============================================================================
#df['ring-number_numeric'] = df['ring-number'].map({'n': 0, 't': 0, 'o': 1})
df['stalk-root_numeric'] = df['stalk-root'].map({'b': 0, '?': 1, 'e': 2, 'c': 3, 'r': 4})
#df['combined-stalk-color'] = df['stalk-color-above-ring'] + '-' + df['stalk-color-below-ring']
df['has-w-color'] = df.apply(lambda row: 1 if 'w' in [row['stalk-color-above-ring'], row['stalk-color-below-ring']] else 0, axis=1)


df['spore-print-color_binary'] = df['spore-print-color'].map({'n': 1, 'k': 1, 'h': 1}).fillna(0).astype(int)
df['cap-color_mapped'] = df['cap-color'].map({'n': 1, 'b': 1, 'w': 1, 'g': 1,
                                              'r': 0, 'p': 0, 'u': 0, 'e': 0, 'y': 0, 'c': 0})


#value_counts = df['stalk-color-above-ring'].value_counts()
#print(value_counts)
# Sonucu göster

# stalk-surface-below-ring dağılımı =================================================================
# s    4936
# k    2304
# f     600
# y     284
# =============================================================================

df['class'] = df['class'].map({'p': 0,'e':1 })
df['population'] = df['population'].map({'a': 1, 'c': 1, 'n': 1, 'v': 1, 's': 0, 'y': 0})
df['habitat'] = df['habitat'].map({'g': 1, 'l': 1, 'm': 1, 'd': 1, 'w': 0, 'u': 0, 'p': 0})
def map_odor(odor):
    if odor == 'n':
        return 0
    elif odor in ['f', 'y', 'm', 'p']:
        return 1
    elif odor in ['a', 'l', 'c', 's']:
        return 2

# Dönüşümü uygulayalım
df['odor_mapped'] = df['odor'].apply(map_odor)

# Sonucu gösterelim
#print(df[['odor', 'odor_mapped']])
# Sonucu görüntüle
#print(df)

# =============================================================================
# =============================================================================
# silinecek_kolonlar=['combined_colors','combined-stalk-surface','stalk-surface-above-ring',
# 'stalk-color-above-ring','odor','spore-print-color','cap-color','stalk-root','stalk-surface-below-ring','stalk-color-below-ring','veil-type','veil-color']              
# for c in silinecek_kolonlar:
#      del df[c]
# =============================================================================
# 
# =============================================================================
df['gill-color_mapped'] = df['gill-color'].map({'k': 1, 'n': 1, 'b': 1, 'h': 1, 'g': 1, 'w': 1,
                                              'r': 0, 'o': 0, 'p': 0, 'u': 0, 'e': 0, 'y': 0})
# 'cap-color' sütununu ve yeni oluşturulan mapped sütunu sil
#df = df.drop(['cap-color', 'cap-color_mapped'], axis=1)

# 'ring-number' sütununu map yapma
df['ring-number_mapped'] = df['ring-number'].map({'n': 0, 'o': 1, 't': 2})

df['cap-color_mapped'] = df['cap-color'].map({'n': 1, 'b': 1, 'w': 1, 'g': 1,
                                              'r': 0, 'p': 0, 'u': 0, 'e': 0, 'y': 0, 'c': 0})
df['cap-surface'] = df['cap-surface'].map({'f': 1, 'g': 2, 'y': 2, 's': 0})

df['ring-type'] = df['ring-type'].map({'p': 1, 'l': 2, 'f': 3, 'e': 0, 'n': 0})



#ring-type sayısına bakılacak
#ring-number n : 0 o : 1 t : 2 map yapıldı silinecek
#cap-color bakıldı silinecek
#gill-color bakıldı silinecek
#cap-surface bakıldı silinecek


silinecek_kolonlar=['combined_colors','combined-stalk-surface','stalk-surface-above-ring',
 'stalk-color-above-ring','odor','spore-print-color','cap-color','stalk-root','stalk-surface-below-ring',
 'stalk-color-below-ring','veil-type','veil-color','gill-color','ring-number']              
for c in silinecek_kolonlar:
      del df[c]


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

aykiri_deger = ['bruises_t','ring-number_mapped',]

for c in aykiri_deger:
  print(aykiri_deger_sil(df,c,0.01))
  print(df.shape)


nan_rows = df[df.isnull().any(axis=1)]

# NaN değerlere sahip sütunları kontrol et
nan_columns = df.columns[df.isnull().any()]

# Sonuçları göster
print("NaN değerlere sahip örnekler:")
print(nan_rows)

print("\nNaN değerlere sahip sütunlar:")
print(nan_columns)
#df["numeric_population"] = df["population"].map({'a':0,'c':1,'n':2})
#df["numeric_population"] = df["population"].map({'a':0,'c':1,'n':2})

"""
# Sadece 'class' ve 'cap-shape' sütunlarını al
selected_columns = ['class', 'cap-shape']
df_selected = df[selected_columns]

# Kategorik değişkenleri sayısala çevirme
df_numeric = df_selected.apply(lambda x: pd.factorize(x)[0])

# Korelasyon matrisini oluşturma
correlation_matrix = df_numeric.corr()

# Heatmap oluşturma
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Korelasyon Matrisi Heatmap")
plt.show()

"""


selected_columns = ['class','cap-shape','cap-surface','ring-type','population'
                ,'habitat','bruises_f','bruises_t','gill-attachment_a',
                'gill-attachment_f','gill-spacing_c','gill-spacing_w','gill-size_b','gill-size_n','stalk-shape_e','stalk-shape_t',
                'veil-type_partial','veil-color_w','combined-stalk-surface-numeric','has-w-color','spore-print-color_binary',
                'cap-color_mapped','odor_mapped','gill-color_mapped','ring-number_mapped',]
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

del df['veil-type_partial']
del df['cap-shape']


#X=df.drop('class',axis=1)
# Yukarıdaki fonksiyonu her satır için uygulayarak yeni bir sütun ekleyebilirsiniz
#df['renk_kombinasyonu'] = df.apply(renk_kombinasyonunu_kontrol_et, axis=1)
df.to_csv('yeni_mushrooms.csv', index=False)



 
