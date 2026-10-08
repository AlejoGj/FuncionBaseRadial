import pandas as pd
import numpy as np

# Cargar el archivo CSV
# Nota: Como usamos punto y coma (;) en la generación anterior, especificamos el separador con sep=';'
df = pd.read_csv('macrodatos_poblacion_colombia-v2.csv', sep=';')



x = df['Año']
y = df['Poblacion_Total']


def crear_matriz(eje_x):
    cantidad_datos = len(eje_x)
    # Creamos una matriz de 20x20 llena de ceros
    matriz_fija = [[0 for _ in range(cantidad_datos)] for _ in range(cantidad_datos)]
    return matriz_fija



def formula_interpolacion(x):
    return np.sqrt(1 + np.square(x))

def matriz_interpolacion(lista_a: list, matriz_vacia):
    for i in range(len(matriz_vacia)):
        for j in range(len(matriz_vacia[0])):

            fi = formula_interpolacion(np.abs(lista_a[i] - lista_a[j]))

            matriz_vacia[i][j] = fi
    
    return matriz_vacia


def calcular_pesos(lista_x, lista_y, matriz_vacia):
    matriz_interpolada = matriz_interpolacion(lista_x, matriz_vacia)

    matriz_inversa = np.linalg.inv(matriz_interpolada)

    return matriz_inversa @ lista_y


# identidad =  np.dot(matriz_fija, matriz_int)

"""print("\nVerificación (Matriz Identidad):")
print(np.round(identidad))"""

#print(matriz_fija)

"""def sumatoria(x, pesos, years, i = 0):
    if i == len(years):
        return 0

    suma = pesos[i] * formula_interpolacion(np.abs(x - years[i]))
    print(suma)
    return suma + sumatoria(x, pesos, years, i + 1)"""


def sumatoria_vectorial(n, pesos, x):
    # np.abs(x - years) calcula todas las distancias al mismo tiempo (vector)
    # formula_interpolacion se aplica a todo el vector
    # np.dot calcula el producto punto (multiplica cada peso por su función y los suma)
    return np.dot(pesos, formula_interpolacion(np.abs(n - x)))





matriz_vacia = crear_matriz(x)

pesos =calcular_pesos(x, y, matriz_vacia)

def calcular_anios(periodo, lista_pred = None, pesosl = pesos, xl = x):
    if lista_pred == None:
        lista_pred = []

    if len(periodo) < 1:
        return lista_pred

    prediccion = sumatoria_vectorial(periodo[0], pesos, xl)
    lista_pred.append(np.round(prediccion, 0))


    return calcular_anios(periodo[1:], lista_pred)


def periodo_prediccion(ini, fin):
    return [n for n in range(ini, fin + 1)]


periodo = periodo_prediccion(2026, 2046)


predicciones = calcular_anios(periodo)

for i in range(len(predicciones)):
    print(f"Prediccion {2026 + i}: ", predicciones[i] )


"""
100     55486605
x       53500000

x = 100(53500000) / 55486605

"""
x=  np.array(x)
y = np.array(y)

lista_anios_full = np.concatenate((x, periodo))
lista_poblacion_full = np.concatenate((y, predicciones))

print(lista_anios_full)
print(lista_poblacion_full)