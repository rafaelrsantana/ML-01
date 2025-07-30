import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


try:
    df = pd.read_excel('diabetes_dataset.xlsx')
    print("Arquivo 'diabetes_dataset.xlsx' carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'diabetes_dataset.xlsx' não encontrado. Verifique se o arquivo está na mesma pasta que o script.")
    exit()


print("--- DataFrame Original com Dados Ausentes ---")
print(df)

imputer = IterativeImputer(max_iter=10, random_state=0)

features = df.drop('Outcome', axis=1)
outcome = df['Outcome']

features_imputed = imputer.fit_transform(features)

df_imputed = pd.DataFrame(features_imputed, columns=features.columns)
df_imputed['Outcome'] = outcome.values

df_imputed.to_excel('diabetes_dataset_tratado.xlsx', index=False)


print("\n--- DataFrame com Dados Ausentes Substituídos pela Mediana ---")
print(df_imputed)