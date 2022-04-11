import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL

engine = create_engine("postgresql://localhost/mydb")
conn = engine.connect()
query= 'SELECT * FROM employees'
pd.read_sql(sql=query, con=conn)
