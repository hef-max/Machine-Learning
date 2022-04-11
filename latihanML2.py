import pandas as pd
import numpy as np

df1 = pd.DataFrame([1,2,None,3,4,5,6,None], columns=["nilai"])
r = np.mean(df1)
df = df1.fillna(r)
print(df)




