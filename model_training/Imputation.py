import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer
from config import DATA_PATH


output_path = '../data/data_imputed.xlsx'

df = pd.read_excel(DATA_PATH, header=0)


df = df.replace(r'^\s*$', np.nan, regex=True)
df = df.replace(['NAN', 'Nan', 'NULL', 'Null', 'null'], np.nan)

missing_cols = df.columns[df.isnull().any()].tolist()

mice_imputer = IterativeImputer(max_iter=20,  initial_strategy='mean')
original_columns = df.columns
imputed_array = mice_imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed_array, columns=original_columns)

df_imputed.to_excel(output_path, index=False)