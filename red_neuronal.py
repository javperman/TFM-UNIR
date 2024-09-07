import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import warnings
warnings.filterwarnings("ignore")

# 1. Importar el df del otro script
import limpieza
df = limpieza.df_transformado

# 2. Separar la variable objetivo ('price') y las variables independientes
X = df.drop('price', axis=1)
y = df['price']

# 3. Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# 4. Escalar las variables numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Definir la función para crear el modelo Keras
def build_model():
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=X_train_scaled.shape[1]))
    model.add(Dropout(0.10))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.10))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 6. Crear el KerasRegressor con EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
keras_regressor = KerasRegressor(build_fn=build_model, epochs=500, batch_size=10, verbose=1, callbacks=[early_stopping])

# 7. Validación cruzada
# Definir las funciones de scoring para cada métrica
scoring = {
    'RMSE': make_scorer(mean_squared_error, squared=False),  # RMSE
    'MAE': make_scorer(mean_absolute_error),                 # MAE
    'R2': make_scorer(r2_score),                             # R2
    'MAPE': make_scorer(lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / y_true)) * 100)  # MAPE
}

# Crear un pipeline que incluya la normalización y el modelo
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', keras_regressor)
])

# Realizar la validación cruzada
cv_results = cross_validate(pipeline, X_train, y_train, cv=10, scoring=scoring)

# Mostrar los resultados
print(f"RMSE scores for each fold: {cv_results['test_RMSE']}")
print(f"Mean RMSE: {cv_results['test_RMSE'].mean():.2f}")
print(f"Standard Deviation of RMSE: {cv_results['test_RMSE'].std():.2f}")

print(f"\nMAE scores for each fold: {cv_results['test_MAE']}")
print(f"Mean MAE: {cv_results['test_MAE'].mean():.2f}")
print(f"Standard Deviation of MAE: {cv_results['test_MAE'].std():.2f}")

print(f"\nR2 scores for each fold: {cv_results['test_R2']}")
print(f"Mean R2: {cv_results['test_R2'].mean():.2f}")
print(f"Standard Deviation of R2: {cv_results['test_R2'].std():.2f}")

print(f"\nMAPE scores for each fold: {cv_results['test_MAPE']}")
print(f"Mean MAPE: {cv_results['test_MAPE'].mean():.2f}%")
print(f"Standard Deviation of MAPE: {cv_results['test_MAPE'].std():.2f}%")

# 8. Entrenar el modelo en todos los datos de entrenamiento con early stopping
keras_regressor.fit(X_train_scaled, y_train)

# 9. Hacer predicciones y evaluar el modelo
y_pred = keras_regressor.predict(X_test_scaled)
df_resultados = pd.DataFrame({'Real': y_test, 
                              'Predicted': y_pred, 
                              'PorcentajeError': abs((y_pred - y_test) / y_test) * 100})
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final Root Mean Squared Error on Test Set: {rmse:.2f}")

import metricas
metricas.metricas(df_resultados['Real'], df_resultados['Predicted'])
