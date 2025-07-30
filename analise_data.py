import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = pd.read_excel('diabetes_dataset.xlsx')
    print("Arquivo 'diabetes_dataset.xlsx' carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'diabetes_dataset.xlsx' não encontrado. Verifique se o arquivo está na mesma pasta que o script.")
    exit()

cols_com_zeros_ausentes = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df_copia = df.copy(deep=True)
df_copia[cols_com_zeros_ausentes] = df_copia[cols_com_zeros_ausentes].replace(0, np.nan)


print("\n--- Análise Estatística Descritiva ---")
estatisticas = df_copia.describe()
print(estatisticas)

print("\nGerando gráficos")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

estatisticas_t = estatisticas.T

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)


estatisticas_t = estatisticas.T
estatisticas_t = estatisticas_t.drop('Outcome')


plt.figure()

ax_mean = sns.barplot(x=estatisticas_t.index, y=estatisticas_t['mean'], palette='viridis')
plt.title('Média das Características', fontsize=16)
plt.ylabel('Valor da Média')
plt.xlabel('Características')
plt.xticks(rotation=45, ha='right')

for bar in ax_mean.patches:
    ax_mean.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')

plt.tight_layout()
plt.show()


plt.figure()
ax_median = sns.barplot(x=estatisticas_t.index, y=estatisticas_t['50%'], palette='plasma')
plt.title('Mediana das Características', fontsize=16)
plt.ylabel('Valor da Mediana')
plt.xlabel('Características')
plt.xticks(rotation=45, ha='right')

for bar in ax_median.patches:
    ax_median.annotate(format(bar.get_height(), '.2f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')

plt.tight_layout()
plt.show()


estatisticas_sem_outcome = estatisticas.drop('Outcome', axis=1)

plt.figure(figsize=(15, 8))
sns.heatmap(estatisticas_sem_outcome, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Mapa de Calor da Tabela de Estatísticas Descritivas', fontsize=16)
plt.show()

missing_data = (df_copia.isnull().sum() / len(df_copia)) * 100

missing_data.sort_values(inplace=True)

plt.figure(figsize=(12, 7))
ax = sns.barplot(x=missing_data.values, y=missing_data.index, orient='h', palette='viridis_r')

plt.title('Porcentagem de Dados Faltantes por Coluna', fontsize=16)
plt.xlabel('Porcentagem (%)', fontsize=12)
plt.ylabel('Características', fontsize=12)

for i, v in enumerate(missing_data):
    ax.text(v + 0.5, i, f'{v:.2f}%', color='black', va='center', fontweight='medium')

plt.xlim(0, 100) 
plt.tight_layout()
plt.show()
