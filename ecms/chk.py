import pandas as pd
df_dbg = pd.read_excel("2.0L SKYACTIV Engine LEV III Fuel_Fuel Map Data.xlsx", header=None)   # 경로는 실제 파일 위치
print(df_dbg.iloc[:12, :8]) 