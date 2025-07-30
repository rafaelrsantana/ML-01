from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.preprocessing import StandardScaler


try:
    df_diabetes = pd.read_excel('diabetes_dataset_tratado.xlsx')
    print("Arquivo 'diabetes_dataset_tratado.xlsx' carregado com sucesso!")

except FileNotFoundError:
    print("Erro: Arquivo 'diabetes_dataset_tratado.xlsx' não encontrado. Verifique se o arquivo está na mesma pasta que o script.")
    exit()
    
try:
    df_diabetes_app = pd.read_excel('diabetes_app.xlsx')
    print("Arquivo 'diabetes_app.xlsx' carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'diabetes_app.xlsx' não encontrado. Verifique se o arquivo está na mesma pasta que o script.")
    exit()

X_treino = df_diabetes.drop('Outcome', axis=1)
y_treino = df_diabetes['Outcome']

X_app = df_diabetes_app

scaler = StandardScaler()

X_treino_escalonado = scaler.fit_transform(X_treino)
X_app_escalonado = scaler.transform(X_app)


df_diabetes_final = pd.DataFrame(X_treino_escalonado, columns=X_treino.columns)
df_diabetes_final['Outcome'] = y_treino.values
df_diabetes_final.to_excel('diabetes_dataset_escalonado.xlsx', index=False)

df_diabetes_app_final = pd.DataFrame(X_app_escalonado, columns=X_app.columns)
df_diabetes_app_final.to_excel('diabetes_app_escalonado.xlsx', index=False)

print("Pré-visualização dos dados escalonados:")
print(df_diabetes_final.head())
print(df_diabetes_app_final.head())