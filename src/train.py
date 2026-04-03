import logging
import pandas as pd
import numpy as np
import joblib
from sklearn import (pipeline, preprocessing, ensemble,feature_selection,
                     model_selection,compose, metrics)

import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('Car Price')
mlflow.sklearn.autolog()

logging.getLogger('mlflow').setLevel(logging.ERROR)

## Dados
print("Carregando dados...")
raw = pd.read_csv('../data/pc_cleaned.csv')

df = raw.copy()
print(f'Foram carregados {raw.shape[0]} linhas e {raw.shape[1]} colunas.')

print("Pré-processando dados...")
# vehicle_age tem multicolinearidade com year
df = df.drop(columns=['vehicle_age']) 

# Split

target = 'price'
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                                                    test_size=0.3, 
                                                                    random_state=42,
                                                                    stratify=X['body_type'])

# Treinamento
## Pipelines
num_pipeline = pipeline.Pipeline([
    ('scaler', preprocessing.StandardScaler()),
])

cat_pipeline = pipeline.Pipeline([
    ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore')),
])

preprocessor = compose.ColumnTransformer(
    transformers=[
        ('num_transformer', num_pipeline, compose.make_column_selector(dtype_include=np.number)),
        ('target_encoder', preprocessing.TargetEncoder(),['model']),
        ('cat_transformer', cat_pipeline, compose.make_column_selector(dtype_include=[object])),
    ]
)

selector = feature_selection.SelectKBest(score_func=feature_selection.mutual_info_regression, 
                                         k=20)

model = ensemble.RandomForestRegressor(random_state=42, 
                                        n_estimators=50,
                                        n_jobs=-1)

pipe = pipeline.Pipeline([
    ('preprocessor',preprocessor),
    ('select_features', selector),
    ('rnd_forest', model),
])

final_pipe = compose.TransformedTargetRegressor(
    regressor=pipe,
    func=np.log1p,
    inverse_func=np.expm1
)

print("Iniciando treinamento...")
with mlflow.start_run(run_name=model.__str__()):
    print('fitting...')
    final_pipe.fit(X_train, y_train)
    print("Fit completo!")
    print("Calculando métricas no treino (cross-validation)...")
    scores = model_selection.cross_val_score(final_pipe, X_train, y_train, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)

    mean_scores = -scores.mean()
    print('Cross-validation completo!')

    print("Avaliando no conjunto de teste...")
    y_pred = final_pipe.predict(X_test)

    rmse = metrics.root_mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    r2 = metrics.r2_score(y_test, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
    print('-Resultados-')
    print(f"Média Scores Validação Cruzada: {mean_scores:.4f}")
    print('MAE_test: ', mae)
    print('RMSE_test: ', rmse)
    print('MAPE_test: ', mape)
    print('R2_test: ', r2)
    print('---')

    mlflow.log_metrics({
    'RMSE' : rmse,
    'MAE' : mae,
    'MAPE' : mape,
    'R2' : r2,
    'mean_scores' : mean_scores,
    })

joblib.dump(final_pipe, '../models/random_forest.pkl')
print("Modelo salvo em ../models/random_forest.pkl")