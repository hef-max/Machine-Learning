import pandas as pd

df = pd.read_csv('dataset/data-jumlah-vaksinasi-remaja-menurut-kelurahan-di-provinsi-dki-jakarta-bulan-juli-tahun-2021.csv')
df2 = df.drop(['tanggal', 'kode_kelurahan'], axis=1)
datakota = df2['wilayah_kota'] == 'JAKARTA PUSAT'
print(df2[datakota])