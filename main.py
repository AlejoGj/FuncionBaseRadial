import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


periodo = periodo_prediccion(2024, 2027)


predicciones = calcular_anios(periodo)

for i in range(len(predicciones)):
    print(f"Prediccion {2026 + i}: ", predicciones[i] )


"""
100     55486605
x       53500000

x = 100(53500000) / 55486605

"""
x = np.array(x)
y = np.array(y)

lista_anios_full = np.concatenate((x, periodo))
lista_poblacion_full = np.concatenate((y, predicciones))

print(lista_anios_full)
print(lista_poblacion_full)


# =====================================================================
# GRAFICACIÓN DE LOS RESULTADOS CON MATPLOTLIB
# =====================================================================
def graficar_interpolacion(anios_reales, poblacion_real, anios_futuros, poblacion_pred, pesos, guardar=True):
    anios_reales = np.array(anios_reales)
    poblacion_real = np.array(poblacion_real)
    anios_futuros = np.array(anios_futuros)
    poblacion_pred = np.array(poblacion_pred)

    plt.figure(figsize=(12, 6))

    # Malla continua para apreciar la curva suave de la función de base radial
    t_hist = np.linspace(anios_reales.min(), anios_reales.max(), 200)
    curva_hist = [sumatoria_vectorial(t, pesos, anios_reales) for t in t_hist]

    t_pred = np.linspace(anios_reales.max(), anios_futuros.max(), 200)
    curva_pred = [sumatoria_vectorial(t, pesos, anios_reales) for t in t_pred]

    # 1. Curva interpolada continua (tramo histórico)
    plt.plot(t_hist, curva_hist, color='#1f77b4', linewidth=2.5, label='Interpolación RBF (2006-2025)')

    # 2. Curva extrapolada continua (tramo proyección)
    plt.plot(t_pred, curva_pred, color='#d62728', linestyle='--', linewidth=2.5, label='Proyección RBF (2026-2046)')

    # 3. Puntos discretos de datos reales
    plt.scatter(anios_reales, poblacion_real, color='#0d47a1', s=50, zorder=5, label='Datos Reales (2006-2025)', edgecolor='black', linewidth=0.6)

    # 4. Puntos discretos predichos año a año
    plt.scatter(anios_futuros, poblacion_pred, color='#b71c1c', marker='s', s=45, zorder=5, label='Predicciones Anuales (2026-2046)', edgecolor='black', linewidth=0.6)

    # 5. Delimitación visual de la zona de proyección
    plt.axvline(x=anios_reales.max(), color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
    plt.axvspan(anios_reales.max(), anios_futuros.max(), color='#ffebee', alpha=0.4, label='Zona de Proyección')

    # 6. Anotaciones de valores clave en los extremos
    plt.annotate(
        f'{poblacion_real[-1] / 1e6:.2f} M ({int(anios_reales[-1])})',
        xy=(anios_reales[-1], poblacion_real[-1]),
        xytext=(-35, 15),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='#0d47a1',
        arrowprops=dict(arrowstyle='->', color='#0d47a1', lw=1.2)
    )

    plt.annotate(
        f'{poblacion_pred[-1] / 1e6:.2f} M ({int(anios_futuros[-1])})',
        xy=(anios_futuros[-1], poblacion_pred[-1]),
        xytext=(-65, 12),
        textcoords='offset points',
        fontsize=9,
        fontweight='bold',
        color='#b71c1c',
        arrowprops=dict(arrowstyle='->', color='#b71c1c', lw=1.2)
    )

    # 7. Títulos, etiquetas y formato de ejes
    plt.title('Interpolación y Proyección de Población en Colombia\nFunción de Base Radial (RBF Multicuádratica)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Año', fontsize=12, fontweight='bold')
    plt.ylabel('Población Total', fontsize=12, fontweight='bold')

    # Formateo del eje Y en millones (M)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y_val, _: f'{y_val * 1e-6:.1f} M'))

    # Configuración de saltos en el eje X
    todos_anios = np.concatenate((anios_reales, anios_futuros))
    plt.xticks(np.arange(todos_anios.min(), todos_anios.max() + 1, 4))

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', frameon=True, framealpha=0.9, shadow=True)
    plt.tight_layout()

    if guardar:
        nombre_archivo = 'grafica_poblacion_rbf.png'
        plt.savefig(nombre_archivo, dpi=300)
        print(f"\n[OK] Grafica guardada exitosamente como '{nombre_archivo}'")

    plt.show()


# Llamada a la graficación
graficar_interpolacion(x, y, periodo, predicciones, pesos)
