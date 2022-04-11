import pandas as pd
import numpy as np


kelompok_usia = [15, 20, 55, 100]
kelompok_usia_label = ['remaja', 'dewasa', 'lanjut_usia']

responded = np.array([17, 18, 19, 20, 21, 26, 17, 18, 20, 23, 35, 80, 70, 40, 50, 60, 90, 17, 19])
binning = pd.cut(responded, kelompok_usia, labels=kelompok_usia_label)
binning = pd.value_counts(binning)
print(binning)
