import pandas as pd
import joblib

# Carrega o modelo
print('Carregando modelo...')
model = joblib.load('../models/random_forest.pkl')
print('Modelo carregado!')

# Carrega os dados
csv_path =  '../data/new_data.csv'
print(f'Carregando dados de {csv_path}...')
df = pd.read_csv(csv_path)

# vehicle_age tem multicolinearidade com year
df = df.drop(columns=['vehicle_age'], errors='ignore')

# Previsões
print('Realizando previsões...')
predictions = model.predict(df)

df['price_pred'] = predictions

if 'price' in df.columns:
    df['diff'] = df['price'] - df['price_pred']
    df['percent_diff'] = df['diff'] * 100 / df['price']
    print(df[['make', 'model', 'price', 'price_pred', 'diff', 'percent_diff']].head(20))
else:
    print(df[['make', 'model', 'price_pred']].head(20))


print('Predições realizadas!')