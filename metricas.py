 # Métricas 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

 # MAE - Mean Absolute Error
def metricas(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Mostrar los resultados 
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")