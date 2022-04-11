import pandas as pd
import numpy as np
import scipy.sparse.csr as csr_matrix
import matplotlib.pyplot as plt
import mlxtend.frequent_patterns as mf
# from apriori_python import apriori

df1 = pd.read_csv('dataset/semua_transaksi_ch11b.csv', parse_dates=['TRX_TS'], index_col=['TRX_ID'])
df1['PRODUCT_NAME'].value_counts().sort_values(ascending=False)

df2 = df1.drop(['PRODUCT_ID'], axis=1).groupby('PRODUCT_NAME').sum()
df2.sort_values(by=['SALES'], ascending=False).head(30).plot(kind='bar')
df1HotEncoded = df1.pivot_table(index='TRX_ID', columns='PRODUCT_NAME', values='SALES').fillna(0)
df1HotEncoded[df1HotEncoded > 0] = 1

df_sparsed = df1HotEncoded.astype(pd.SparseDtype('float', np.nan))

df3 = mf.apriori(df_sparsed, min_support=0.1, use_colnames=True)

df4 =  mf.association_rules(df3, metric='lift', min_threshold=1)
df4 = df4.sort_values(by=['support', 'confidence', 'lift'], ascending=False)
print(df4)


