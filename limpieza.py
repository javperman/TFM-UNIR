import pandas as pd
import ast

# FUNCIONES -------------------------------------------------------------------
# Función para verificar si 'labels' contiene {'name': 'luxuryType', 'text': 'Lujo'}
def contains_luxury(label):
    try:
        labels_list = ast.literal_eval(label)
        for item in labels_list:
            if item == {'name': 'luxuryType', 'text': 'Lujo'}:
                return True
    except (ValueError, SyntaxError):
        return False
    return False

# Función para verificar si 'parkingSpace' contiene {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
def contains_parking(label):
    if pd.isna(label):
        return False
    try:
        # Convertir la cadena en un diccionario
        label_dict = ast.literal_eval(label)
        
        # Verificar que sea un diccionario y que las claves existan y sean True
        return label_dict.get('hasParkingSpace') is True and label_dict.get('isParkingSpaceIncludedInPrice') is True
    except (ValueError, SyntaxError):
        # En caso de que la conversión falle, devolver False
        return False
#------------------------------------------------------------------------------

# Leer el csv y guardarlo en un DataFrame
file_path = 'idealistaAlquiler.csv'
df = pd.read_csv(file_path)

# Vemos los distintos valores de municipality y nos quedamos solo con Sevilla
distinct_municipalities = df['municipality'].unique()
distinct_municipalities = sorted(distinct_municipalities)
df = df[df['municipality'] == 'Sevilla']
df = df[sorted(df.columns)]

# Leer el csv de compra y guardarlo en un DataFrame
file_path_compra = 'idealistaCompraCorregido.csv'
df_compra = pd.read_csv(file_path_compra)

# Variables ordenadas alfabéticamente
sorted_columns = sorted(df.columns)
sorted_columns_compra = sorted(df_compra.columns)
print('Las columnas de alquiler y compra son las mismas? --> ', sorted_columns==sorted_columns_compra)

# Nos quedamos solo con las variables que van a alimentar al modelo
columnas = ['bathrooms',
            #'district',
            'exterior',
            'floor',
            'hasLift',
            'labels',
            'latitude', 
            'longitude',
            #neighborhood,
            'newDevelopment',
            'parkingSpace', 
            'price', 
            'propertyType', 
            'rooms', 
            'size', 
            'status']
df = df[columnas]

# Ver si hay duplicados y eliminarlos
duplicados = df.duplicated()
print(f"Hay {duplicados.sum()} registros duplicados en df. Se muestran en el df_duplicados")
df_duplicados = df[duplicados]
df = df.drop_duplicates()
print("Se han eliminado los duplicados")

#------------------------------------------------------------------------------
# Analizar los valores que toman todas las columnas
column_analysis = {}

for column in df.columns:
    unique_values = sorted(df[column].astype(str).unique())
    value_counts = df[column].value_counts()
    column_summary = df[column].describe()
    null_values = df[column].isna().sum()
    
    column_analysis[column] = {
        'unique_values': unique_values,
        'value_counts': value_counts,
        'summary': column_summary,
        'null_values': null_values
    }

# Mostrar el análisis de las columnas
for column, analysis in column_analysis.items():
    print(f"Análisis de la columna {column}")
    print("Valores únicos:")
    print(analysis['unique_values'])
    print("\nConteo de valores:")
    print(analysis['value_counts'])
    print("\nResumen:")
    print(analysis['summary'])
    print(f"\nValores nulos: {analysis['null_values']}")
    print("\n" + "-"*50 + "\n")
#------------------------------------------------------------------------------

## Explicación de las transformaciones de las variables en el excel

# BATHROOMS	integer	Dejar como numérica

# EXTERIOR	boolean	SÍ	Categórica teniendo en cuenta el nan
# Crear nuevas columnas para la codificación categórica binaria
df['exterior_0'] = 0
df['exterior_1'] = 0
# Mapear los valores a la codificación binaria
df.loc[df['exterior'] == True, ['exterior_0', 'exterior_1']] = [0, 1]
df.loc[df['exterior'] == False, ['exterior_0', 'exterior_1']] = [1, 0]
df.loc[df['exterior'].isna(), ['exterior_0', 'exterior_1']] = [0, 0]

# FLOOR categórica. Antes quitamos los valores con floor nan y que no sean chalets
df = df[~(df['floor'].isna() & (df['propertyType'] != 'chalet'))]
#df['floor_0'] = 0 Si todo es 0 es porque es bj o ss
df['floor_1'] = 0
df['floor_2'] = 0
df['floor_3'] = 0
df['floor_4'] = 0
df['floor_high'] = 0
df['floor_en'] = 0
# Mapear los valores
df.loc[df['floor'] == '1', 'floor_1'] = 1
df.loc[df['floor'] == '2', 'floor_2'] = 1
df.loc[df['floor'] == '3', 'floor_3'] = 1
df.loc[df['floor'] == '4', 'floor_4'] = 1
df.loc[df['floor'] == 'en', 'floor_en'] = 1
df.loc[pd.to_numeric(df['floor'], errors='coerce') >= 5, 'floor_high'] = 1

# HASLIFT Asignar True a los que tengan floor>=5
df.loc[(df['floor_high'] == 1) & (df['hasLift'].isna()), 'hasLift'] = True
df.loc[(df['floor_high'] == 0) & (df['hasLift'].isna()), 'hasLift'] = False

# LABELS Crear columna LUXURY con valores True y False 
df['luxury'] = False
df['luxury'] = df['labels'].apply(lambda x: contains_luxury(x))

# LATITUDE Dejar como numérica

# LONGITUDE Dejar como numérica

# NEWDEVELOPMENT Dejar como categórica True False

# PARKINGSPACE Crear columna parking True False
df['hasParking'] = False
df['hasParking'] = df['parkingSpace'].apply(lambda x: contains_parking(x))

# PRICE Dejar como numérica

# PROPERTYTYPE categórica ['chalet', 'duplex', 'flat', 'penthouse', 'studio']
df['property_chalet'] = 0
df['property_duplex'] = 0
#df['property_flat'] = 0 Si todo es 0 es un flat
df['property_penthouse'] = 0
df['property_studio'] = 0
# Mapear los valores
df.loc[df['propertyType'] == 'chalet', 'property_chalet'] = 1
df.loc[df['propertyType'] == 'duplex', 'property_duplex'] = 1
df.loc[df['propertyType'] == 'penthouse', 'property_penthouse'] = 1
df.loc[df['propertyType'] == 'studio', 'property_studio'] = 1

# ROOMS dejar como numérica

# SIZE dejar como numérica

# STATUS categórica ['good' (0,0), 'newdevelopment' (1,0), 'renew' (0,1)]
#df['status_good'] = 0 Si es good es 0
#df['status_new'] = 0 YA TENEMOS ESTA INFO EN LA VARIABLE NEWDEVELOPMENT
df['status_renew'] = 0
# Mapear los valores
df.loc[df['status'] == 'renew', 'status_renew'] = 1
 
# Eliminar las columnas originales necesarias
df_transformado = df.drop(columns=['exterior', 'floor', 'labels', 'parkingSpace', 'propertyType', 'status'], inplace=False)
