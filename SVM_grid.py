import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import make_scorer, mean_squared_error

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

# 5. Crear el modelo de SVM para regresión (SVR)
svr_regressor = SVR()

# GRID SEARCH
# Definir el espacio de hiperparámetros para la búsqueda
param_grid = {
    'kernel' : ['rbf'],
    'C' : [100,1000,10000],
    'gamma' : ['auto','scale']
}
 
# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=svr_regressor, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
 
# Ejecutar GridSearchCV
grid_search.fit(X_train, y_train)
 
# Mejor combinación de hiperparámetros encontrada
print("Mejores hiperparámetros:", grid_search.best_params_)

# Mejor modelo entrenado
svr_best = grid_search.best_estimator_

# 6. Validación cruzada
rmse_scorer = make_scorer(mean_squared_error, squared=False)  # Definir el RMSE como métrica
cv_scores = cross_val_score(svr_best, X_train_scaled, y_train, cv=10, scoring=rmse_scorer)

# Mostrar resultados de la validación cruzada
print(f"RMSE scores for each fold: {cv_scores}")
print(f"Mean RMSE: {cv_scores.mean():.2f}")
print(f"Standard Deviation of RMSE: {cv_scores.std():.2f}")

# 7. Entrenar el modelo en todos los datos de entrenamiento
svr_best.fit(X_train_scaled, y_train)

# 8. Hacer predicciones y evaluar el modelo
y_pred = svr_best.predict(X_test_scaled)
df_resultados = pd.DataFrame({'Real': y_test, 
                              'Predicted': y_pred, 
                              'PorcentajeError': abs((y_pred - y_test) / y_test) * 100})
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Final Root Mean Squared Error on Test Set: {rmse:.2f}")

import metricas
metricas.metricas(df_resultados['Real'], df_resultados['Predicted'])