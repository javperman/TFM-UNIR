from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 0. Importar el df de compra del otro script y el modelo a usar
import limpieza_compra
import random_forest_best #Modelo a usar
df = limpieza_compra.df_transformado

# 1. Separar la variable price (de compra) para guardarla y hacer cálculos de rentabilidad
precios_compra = df['price'].to_list()

# 2. Quedarnos con los datos sin la variable price para alimentar al modelo
X = df.drop('price', axis=1)

# 3. Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Realizar las predicciones
y_pred = random_forest_best.random_forest_regressor.predict(X_scaled)
y_pred_df = pd.DataFrame(y_pred, columns=['y_pred'])

# 5. Calcular las rentabilidades
df_con_predicciones = pd.concat([df, y_pred_df], axis=1)
df_con_predicciones['rentabilidad'] = (df_con_predicciones['y_pred'] * 12) / df_con_predicciones['price'] * 100

# 6. Nos quedamos solo con las que cuestan al menos 70000 para evitar viviendas ocupadas o situaciones especiales
df_con_predicciones_filtrado = df_con_predicciones[df_con_predicciones['price']>=70000] # Observamos como las rentabilidades son ahora más adecuadas.

# ANÁLISIS EXPLORATORIO
# Descriptivo variables
descrip = df_con_predicciones_filtrado.describe()
rentabilidad_media = descrip['rentabilidad']['mean']
alquiler_medio = descrip['y_pred']['mean']
print(f'Rentabilidad anual media del {rentabilidad_media:.2f}%')
print(f'Alquiler medio de {alquiler_medio:.2f}€')

#Distribución rentabilidad
df_con_predicciones_filtrado['rentabilidad'].hist(bins=30)
plt.title('Distribución de la Rentabilidad')
plt.xlabel('Rentabilidad anual')
plt.ylabel('Número de inmuebles')
plt.show()


# Mapa de calor correlación
plt.figure(figsize=(20, 16))
heatmap = sns.heatmap(df_con_predicciones_filtrado.corr(), 
                      annot=True, 
                      cmap='coolwarm', 
                      fmt=".2f",          # Limitar los números a 2 decimales
                      annot_kws={"size": 8},  # Reducir el tamaño de las anotaciones
                      linewidths=.5)      # Añadir espacio entre celdas
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize=10)
plt.title('Mapa de Calor de Correlaciones', fontsize=18)  # Aumentar el tamaño del título
plt.show()

# Mapa de calor de las variables con mayor correlacion
columnas_interes = ['rentabilidad', 'bathrooms', 'price', 'rooms']
df_subset = df_con_predicciones_filtrado[columnas_interes]
correlation_matrix = df_subset.corr()
plt.figure(figsize=(10, 8))  # Ajustar tamaño según sea necesario
heatmap = sns.heatmap(correlation_matrix, 
                      annot=True, 
                      cmap='coolwarm', 
                      fmt=".2f",          # Limitar los números a 2 decimales
                      annot_kws={"size": 10},  # Ajustar el tamaño de las anotaciones
                      linewidths=.5)      # Añadir espacio entre celdas
plt.title('Mapa de Calor de Correlaciones para Variables Seleccionadas', fontsize=16)
plt.show()

#Análisis de las variables categóricas y rentabilidad
variables_categoricas = ['hasLift', 'newDevelopment', 'luxury', 'hasParking', 'status_renew']
for var in variables_categoricas:
    sns.boxplot(x=var, y='rentabilidad', data=df_con_predicciones_filtrado)
    plt.title(f'Rentabilidad vs {var}')
    plt.show()

for var in variables_categoricas:
    print(df_con_predicciones_filtrado.groupby(var)['rentabilidad'].mean())
    

# Análisis de las variables por Tipo de Inmueble
df_con_predicciones_filtrado['property_type'] = 'Piso'  # Asignar valor por defecto

df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['property_chalet'] == 1, 'property_type'] = 'Chalet'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['property_duplex'] == 1, 'property_type'] = 'Dúplex'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['property_penthouse'] == 1, 'property_type'] = 'Ático'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['property_studio'] == 1, 'property_type'] = 'Estudio'

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_con_predicciones_filtrado, x='property_type', y='rentabilidad', palette='Set2')


plt.title('Distribución de la Rentabilidad por Tipo de Propiedad', fontsize=16)
plt.xlabel('Tipo de Propiedad', fontsize=14)
plt.ylabel('Rentabilidad', fontsize=14)
plt.xticks(rotation=45)  # Rotar etiquetas del eje x si es necesario
plt.show()


# Análisis de las variables por Altura del inmueble
df_con_predicciones_filtrado['floor_type'] = 'Bajo/Semisótano'  # Asignar valor por defecto

df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['floor_1'] == 1, 'floor_type'] = 'Primer Piso'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['floor_2'] == 1, 'floor_type'] = 'Segundo Piso'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['floor_3'] == 1, 'floor_type'] = 'Tercer Piso'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['floor_4'] == 1, 'floor_type'] = 'Cuarto Piso'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['floor_high'] == 1, 'floor_type'] = 'Piso Alto'
df_con_predicciones_filtrado.loc[df_con_predicciones_filtrado['floor_en'] == 1, 'floor_type'] = 'Entrepiso'

plt.figure(figsize=(12, 8))
sns.boxplot(data=df_con_predicciones_filtrado, x='floor_type', y='rentabilidad', palette='Set2')

plt.title('Distribución de la Rentabilidad por Altura Inmueble', fontsize=16)
plt.xlabel('Tipo de Piso', fontsize=14)
plt.ylabel('Rentabilidad', fontsize=14)
plt.xticks(rotation=45)  # Rotar etiquetas del eje x si es necesario
plt.show()

df_con_predicciones_filtrado.to_csv('dataset_predicciones_filtrado.csv', index=False, sep=';', decimal='.')