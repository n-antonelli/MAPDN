import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import time
import ast
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


mode = 'eval'  # train/eval
topology = 'case33_3min_final'
clase = 'distributed'  # distributed/decentralised
algoritmo = "mappo"
loss_type= 'bowl'
nombre_modelo = f'var_voltage_control-{topology}-{clase}-{algoritmo}-{loss_type}-original'

# entrenamiento
evento = '1791370814'
# evaluación
dia = 'day880' #'batch'/'day900'

# actual_date = datetime.now().date()
# ventana = 50
# device = 'oficina'
if mode == 'train':
    carpeta = 'train_save'
else:
    carpeta = 'eval_save'
url = f"C:\\PDN_runs\\MAPDN\\{carpeta}"
# url = "C:\\Users\\Usuario\\Documents\\Programas\MAPDN\\tensorboard"
# if device == 'oficina':
#
# # else:
# #     url = 'C:\\Users\\Nicolas\\Documents\\UNSL\\Programas\\MARLlib-PDN'

steps = np.arange(480)
time_index = pd.date_range(start="00:00", periods=480, freq="3min")
time_labels = time_index.strftime("%H:%M")
tick_positions = np.arange(0, 480, 40)  # cada 48 steps = cada 2.4 hs ~ cada 2hs

def calcular_promedio_movil(datos, ventana, mode):
    """
    Calcula el promedio móvil.
    En los bordes donde no cabe la ventana, MANTIENE los datos originales.
    """
    datos = np.asarray(datos)
    n = len(datos)

    # Si hay menos datos que la ventana, devolvemos tal cual
    if n < ventana:
        return datos

    # 1. Crear el kernel de promedio
    ventana_kernel = np.ones(ventana) / ventana

    # 2. Calcular la parte central suavizada ('valid')
    # Esto reduce el tamaño del array
    datos_validos = np.convolve(datos, ventana_kernel, mode='valid')

    # 3. Calcular el offset (cuántos datos quedan a la izquierda)
    # Al centrar la ventana, sobran 'ventana // 2' elementos a cada lado aprox.
    offset = ventana // 2

    # 4. Construir el resultado final
    # a) Inicio: Tomamos los datos ORIGINALES desde el 0 hasta el offset
    parte_inicio = datos[:offset]

    # b) Final: Tomamos los datos ORIGINALES desde el final de la parte válida
    # Calculamos dónde termina la parte válida dentro del array original
    indice_final = offset + len(datos_validos)
    if mode == 'eval':
        parte_final = datos[indice_final:]
    else:
        parte_final = datos[indice_final: -ventana]

    # c) Concatenar: Originales + Promedio + Originales
    datos_suavizados = np.concatenate([parte_inicio, datos_validos, parte_final])

    return datos_suavizados

init = 0
final = 6200
cantidad_valores_mostrados = 100

if mode == 'train':
    ruta = f'{url}\\{nombre_modelo}\\events.out.tfevents.{evento}.DESKTOP-I6IL4F1'
    train_archive = EventAccumulator(ruta)
    train_archive.Reload()
    # print(train_archive.Tags()["scalars"])

    df = pd.DataFrame({
        tag.removeprefix("data/"): pd.Series(
            [e.value for e in train_archive.Scalars(tag)],
            index=[e.step for e in train_archive.Scalars(tag)]
        )
        for tag in train_archive.Tags()["scalars"]
    })
    folder = Path(r'C:\PDN_runs\Pruebas\results')
    # cantidad_agentes = len(train_archive['pv_active'])
    # agents = [f'agent_{i + 1}' for i in range(cantidad_agentes)]

    df[["mean_train_reward"]].plot()
    plt.xlabel("episode")
    plt.ylabel("mean_train_reward")
    plt.show()
    df[["mean_test_reward"]].plot()
    plt.xlabel("episode")
    plt.ylabel("mean_test_reward")
    plt.show()

    df[["mean_train_total_line_loss"]].plot()
    plt.xlabel("episode")
    plt.ylabel("mean_train_total_line_loss")
    plt.show()
    df[["mean_train_q_loss"]].plot()
    plt.xlabel("episode")
    plt.ylabel("mean_train_q_loss")
    plt.show()
    # print(df["mean_test_reward"]¿)
    #
    # figli, axli = plt.subplots(2, 1, figsize=(12, 6))
    # axli[0].set_title("bus_voltage")
    # axli[1].set_title("Potencia generada")
    # axli1 = axli[1].twinx()
    # axli[0].plot(range(0, len(np.asarray(test_archive["bus_voltage"]))), np.asarray(test_archive["bus_voltage"]), label='bus_voltage')
    # axli[0].axhline(y=0.95, color='red', linestyle='--', linewidth=1)
    # axli[0].axhline(y=1.05, color='red', linestyle='--', linewidth=1)
    # axli[1].plot(range(0, len(np.asarray(test_archive["pv_active"]))), np.asarray(test_archive["pv_active"]), label='active', color='blue')
    # axli1.plot(range(0, len(np.asarray(test_archive["pv_reactive"]))), np.asarray(test_archive["pv_reactive"]), label='reactive', color='orange')
    # # axli[1].plot(range(0, len(np.asarray(ep["va_degree_bus"]))), np.asarray(ep["va_degree_bus"]), label='va_degree_bus')
    # axli[0].set_xticks([])
    # axli[1].set_xticks([])
    # # axli[0].legend(loc='best')
    # axli[1].legend(loc='upper left')
    # axli[1].set_ylabel('pv_active(blue)')
    # axli1.legend(loc='upper right')
    # axli1.set_ylabel('pv_reactive(orange)')
    # figli.show()
    #
    # figlo, axlo = plt.subplots(2, 1, figsize=(12, 6))
    # axlo[0].set_title('line_loss')
    # axlo[1].set_title('bus_active/reactive')
    # axlo1 = axlo[1].twinx()
    # axlo[0].plot(range(0, len(np.asarray(test_archive["line_loss"]))), np.asarray(test_archive["line_loss"]), label='line_loss')
    # axlo[1].plot(range(0, len(np.asarray(test_archive["bus_active"]))), np.asarray(test_archive["bus_active"]), label='bus_active', color='blue')
    # axlo1.plot(range(0, len(np.asarray(test_archive["bus_reactive"]))), np.asarray(test_archive["bus_reactive"]), label='bus_reactive', color='orange')
    # # axlo[0].legend(loc='best')
    # # axlo[1].legend(loc='upper left')
    # # axlo1.legend(loc='upper right')
    # axlo[1].set_ylabel('bus_active(blue)')
    # axlo1.set_ylabel('bus_reactive(orange)')
    # axlo[1].set_xticks([])
    # figlo.show()

elif mode == 'eval':
    archivo = f"test_{nombre_modelo}_{dia}.pickle"
    with open(f'{url}\\{archivo}', "rb") as f:
        test_archive = pickle.load(f)
    ## sacar pimeros 3 steps
    test_archive = {clave: valor[3:] for clave, valor in test_archive.items()}
    # cantidad_agentes = len(test_archive['pv_active'])
    # agents = [f'agent_{i + 1}' for i in range(cantidad_agentes)]
    # print(test_archive)
    figli, axli = plt.subplots(2, 1, figsize=(12, 6))
    axli[0].set_title("bus_voltage")
    axli[1].set_title("Potencia generada")
    axli1 = axli[1].twinx()
    axli[0].plot(range(0, len(np.asarray(test_archive["bus_voltage"]))), np.asarray(test_archive["bus_voltage"]), label='bus_voltage')
    axli[0].axhline(y=0.95, color='red', linestyle='--', linewidth=1)
    axli[0].axhline(y=1.05, color='red', linestyle='--', linewidth=1)
    axli[1].plot(range(0, len(np.asarray(test_archive["pv_active"]))), np.asarray(test_archive["pv_active"]), label='active', color='blue')
    axli1.plot(range(0, len(np.asarray(test_archive["pv_reactive"]))), np.asarray(test_archive["pv_reactive"]), label='reactive', color='orange')
    # axli[1].plot(range(0, len(np.asarray(ep["va_degree_bus"]))), np.asarray(ep["va_degree_bus"]), label='va_degree_bus')
    axli[0].set_xticks([])
    axli[1].set_xticks([])
    # axli[0].legend(loc='best')
    axli[1].legend(loc='upper left')
    axli[1].set_ylabel('pv_active(blue)')
    axli1.legend(loc='upper right')
    axli1.set_ylabel('pv_reactive(orange)')
    figli.show()

    figlo, axlo = plt.subplots(2, 1, figsize=(12, 6))
    axlo[0].set_title('line_loss')
    axlo[1].set_title('bus_active/reactive')
    axlo1 = axlo[1].twinx()
    axlo[0].plot(range(0, len(np.asarray(test_archive["line_loss"]))), np.asarray(test_archive["line_loss"]), label='line_loss')
    axlo[1].plot(range(0, len(np.asarray(test_archive["bus_active"]))), np.asarray(test_archive["bus_active"]), label='bus_active', color='blue')
    axlo1.plot(range(0, len(np.asarray(test_archive["bus_reactive"]))), np.asarray(test_archive["bus_reactive"]), label='bus_reactive', color='orange')
    # axlo[0].legend(loc='best')
    # axlo[1].legend(loc='upper left')
    # axlo1.legend(loc='upper right')
    axlo[1].set_ylabel('bus_active(blue)')
    axlo1.set_ylabel('bus_reactive(orange)')
    axlo[1].set_xticks([])
    figlo.show()
