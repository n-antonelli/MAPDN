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
import glob
import os
import re


mode = 'train'  # train/test
topology = 'case33_3min_final'
clase = 'decentralised'  # distributed/decentralised
algoritmo = "mappo" # maddpg/mappo
loss_type = 'bowl'
alias = 'get_actions-batch-episodic-returns-shared-lrate-sin_norm'
nombre_modelo = f'var_voltage_control-{topology}-{clase}-{algoritmo}-{loss_type}-{alias}'

if mode == 'train':
    carpeta = 'train_save'
else:
    carpeta = 'eval_save'
url = f"C:\\PDN_runs\\MAPDN\\{carpeta}"
# url = "C:\\Users\\Usuario\\Documents\\Programas\MAPDN\\tensorboard"
# if device == 'oficina':
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
    # if mode == 'eval':
    parte_final = datos[indice_final:]
    # else:
    #     parte_final = datos[indice_final: -ventana]

    # c) Concatenar: Originales + Promedio + Originales
    datos_suavizados = np.concatenate([parte_inicio, datos_validos, parte_final])

    return datos_suavizados

init = 0
final = 6200
cantidad_valores_mostrados = 100

if mode == 'train':
    # ruta = f'{url}\\{nombre_modelo}\\events.out.tfevents.{evento}.DESKTOP-I6IL4F1'
    patron = os.path.join(url, nombre_modelo, 'events.out.tfevents.*')
    ruta = glob.glob(patron)[0]
    train_archive = EventAccumulator(ruta)
    train_archive.Reload()
    print(train_archive.Tags()["scalars"])

    df = pd.DataFrame({
        tag.removeprefix("data/"): pd.Series(
            [e.value for e in train_archive.Scalars(tag)],
            index=[e.step for e in train_archive.Scalars(tag)]
        )
        for tag in train_archive.Tags()["scalars"]
    })
    folder = Path(r'C:\PDN_runs\Pruebas\results')
    # print(df.keys())
    # cantidad_agentes = len(train_archive['pv_active'])
    # agents = [f'agent_{i + 1}' for i in range(cantidad_agentes)]

    mean_train_reward = calcular_promedio_movil(df["mean_train_reward"], 20, mode)#
    test_clean = df['mean_test_reward'].dropna()
    plt.plot(mean_train_reward, label='mean_train_reward', alpha=0.35, color='gray')
    plt.plot(test_clean.index, test_clean, marker='o', linestyle='-', color='tab:blue', label='mean_test_reward')
    plt.title(f"{algoritmo}-{clase} - mean train test reward")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()

    mean_train_totally_cr = calcular_promedio_movil(df["mean_train_totally_controllable_ratio"], 20, mode)  #
    test_clean = df['mean_test_totally_controllable_ratio'].dropna()
    plt.plot(mean_train_totally_cr, label='mean_train_totally_cr', alpha=0.35, color='gray')
    plt.plot(test_clean.index, test_clean, marker='o', linestyle='-', color='tab:blue', label='mean_test_totally_cr')
    plt.title(f"{algoritmo}-{clase} - mean train/test totally controllable ratio")
    plt.grid(True, alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("mean_train/test_totally_cr")
    plt.legend(loc='best')
    plt.show()

    mean_train_avg_line_loss = calcular_promedio_movil(df["mean_train_avg_line_loss"], 20, mode)  #
    test_clean = df['mean_test_avg_line_loss'].dropna()
    plt.plot(mean_train_avg_line_loss, label='mean_train_avg_line_loss', alpha=0.35, color='gray')
    plt.plot(test_clean.index, test_clean, marker='o', linestyle='-', color='tab:blue', label='mean_test_avg_line_loss')
    plt.title(f"{algoritmo}-{clase} - mean train/test avg line loss")
    plt.grid(True, alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("mean_train/test_avg_line_loss")
    plt.legend(loc='best')
    plt.show()

    mean_train_percentage_of_v_ooc = calcular_promedio_movil(df["mean_train_percentage_of_v_out_of_control"], 20, mode)  #
    test_clean = df['mean_test_percentage_of_v_out_of_control'].dropna()
    plt.plot(mean_train_percentage_of_v_ooc, label='mean_train_percentage_of_v_ooc', alpha=0.35, color='gray')
    plt.plot(test_clean.index, test_clean, marker='o', linestyle='-', color='tab:blue', label='mean_test_percentage_of_v_ooc')
    plt.title(f"{algoritmo}-{clase} - mean train/test totally v out of control")
    plt.grid(True, alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("mean_train_percentage_of_v_out_of_control")
    plt.show()

    df["mean_train_value_loss"] = calcular_promedio_movil(df["mean_train_value_loss"], 20, mode)
    mean_train_value_loss = df["mean_train_value_loss"]
    test_clean = df['mean_train_destroy']*3
    plt.plot(mean_train_value_loss, label='mean_train_value_loss')
    plt.plot(test_clean.index, test_clean, linestyle='-', color='orange', label='destroy')
    plt.title(f"{algoritmo}-{clase} - mean train value loss")
    plt.xlabel("episode")
    plt.ylabel("mean_train_value_loss")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.show()

    df["mean_train_ev"] = calcular_promedio_movil(df["mean_train_ev"], 20, mode)
    df[["mean_train_ev"]].plot()
    plt.title(f"{algoritmo}-{clase} - mean train ev")
    plt.grid(True, alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("mean_train_ev")
    plt.show()

    df["mean_train_policy_loss"] = calcular_promedio_movil(df["mean_train_policy_loss"], 20, mode)
    df[["mean_train_policy_loss"]].plot()
    plt.title(f"{algoritmo}-{clase} - mean train policy loss")
    plt.grid(True, alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("mean_train_policy_loss")
    plt.show()

    df["mean_train_policy_grad_norm"] = calcular_promedio_movil(df["mean_train_policy_grad_norm"], 20, mode)
    df[["mean_train_policy_grad_norm"]].plot()
    plt.title(f"{algoritmo}-{clase} - mean train policy grad norm")
    plt.grid(True, alpha=0.3)
    plt.xlabel("episode")
    plt.ylabel("mean_train_policy_grad_norm")
    plt.show()


elif mode == 'test':
    # archivo = f"test_{nombre_modelo}_{dia}.pickle"
    patron = os.path.join(url, f"test_{nombre_modelo}_*.pickle")
    archivos = glob.glob(patron)
    for ruta in archivos:
        with open(ruta, "rb") as f:
            test_archive = pickle.load(f)
        ## sacar pimeros 3 steps
        # test_archive = {clave: valor[3:] for clave, valor in test_archive.items()}
        # cantidad_agentes = len(test_archive['pv_active'])
        # agents = [f'agent_{i + 1}' for i in range(cantidad_agentes)]
        # print(test_archive)
        match = re.search(r"day(\d+)", ruta)
        dia = match.group(1)
        figli, axli = plt.subplots(2, 1, figsize=(12, 6))
        figli.suptitle(f'{algoritmo}-{clase}')
        axli[0].set_title(f"bus_voltage-día {dia}")
        axli[1].set_title(f"line_loss-día {dia}")
        axli[0].plot(range(0, len(np.asarray(test_archive["bus_voltage"]))), np.asarray(test_archive["bus_voltage"]), label='bus_voltage')
        axli[0].axhline(y=0.95, color='red', linestyle='--', linewidth=1)
        axli[0].axhline(y=1.05, color='red', linestyle='--', linewidth=1)
        axli[1].plot(range(0, len(np.asarray(test_archive["line_loss"]))), np.asarray(test_archive["line_loss"]), label='line_loss')
        axli[0].set_xticks([])
        axli[1].set_xticks([])
        axli[0].grid(True, alpha=0.3)
        axli[1].grid(True, alpha=0.3)
        # axli[0].legend(loc='best')
        # axli[1].legend(loc='upper left')
        figli.show()

        figlo, axlo = plt.subplots(1, 1, figsize=(12, 6))
        figlo.suptitle(f'{algoritmo}-{clase}')
        axlo.set_title(f'v_loss-día {dia}')
        # axlo[1].set_title('q_loss')
        # axlo0 = axlo[0].twinx()
        axlo.plot(range(0, len(np.asarray(test_archive["v_loss"]))), np.asarray(test_archive["v_loss"]), label='v_loss', color='blue')
        # axlo0.plot(range(0, len(np.asarray(test_archive["pv_reactive"]))), np.asarray(test_archive["pv_reactive"]), label='reactive', color='orange')
        axlo.plot(range(0, len(np.asarray(test_archive["q_loss"]))), np.asarray(test_archive["q_loss"]), label='q_loss', color='orange')
        axlo.plot(range(0, len(np.asarray(test_archive["avg_line_loss"]))), np.asarray(test_archive["avg_line_loss"]), label='avg_line_loss', color='red')
        # axlo[1].plot(range(0, len(np.asarray(test_archive["actual_actions"]))), np.asarray(test_archive["actual_actions"]), label='actual_actions', color='orange')
        axlo.legend(loc='upper left')
        axlo.grid(True, alpha=0.3)
        # axlo[0].set_ylabel('pv_active(blue)')
        # axlo0.legend(loc='upper right')
        # axlo0.set_ylabel('pv_reactive(orange)')
        # axlo[1].legend(loc='upper left')
        # axlo[1].set_xticks([])
        figlo.show()


        def plot_agents(archive, step_min=3, act_lim=0.8):
            P = np.asarray(archive["pv_active"][1:])  # (T, 6)
            Q = np.asarray(archive["pv_reactive"][1:])  # (T, 6)
            A = np.asarray(archive["actions"])  # (T, 6)  crudas
            Aa = np.asarray(archive["actual_actions"])  # (T, 6)  aplicadas
            T, n = P.shape
            h = np.arange(T) * step_min / 60  # horas del día
            q_lim = np.abs(Q).max() * 1.1
            n = test_archive["n_agent"]
            fig, axs = plt.subplots(n, 2, figsize=(13, 2.0 * n), sharex=True)
            fig.suptitle(f'{algoritmo}-{clase}')

            if clase=='distributed':
                pv_p_agente = ['zona1-n13', 'zona1-n18', 'zona2-n22', 'zona3-n25', 'zona4-n29', 'zona4-n33']
            else:
                pv_p_agente = ['zona1-n13/n18', 'zona2-n22', 'zona3-n25', 'zona4-n29/n33']

            for i in range(n):
                # Izquierda: P (área, eje izq.) y Q (línea, eje der.)
                ax = axs[i, 0]
                ax.fill_between(h, P[:, i], color="tab:blue", alpha=0.35, lw=0)
                ax.set_ylabel(f"Agente {i}\nP [MW] \n {pv_p_agente[i]}", color="tab:blue")
                axq = ax.twinx()
                axq.plot(h, Q[:, i], color="tab:orange", lw=1.3)
                axq.axhline(0, color="k", lw=0.4)
                axq.set_ylim(-q_lim, q_lim)
                axq.set_ylabel("Q [MVAR]", color="tab:orange")

                # Derecha: acciones
                ax = axs[i, 1]
                ax.plot(h, A[:, i], color="gray", ls="--", lw=1, label="cruda")
                ax.plot(h, Aa[:, i], color="tab:green", lw=1.4, label="aplicada")
                ax.axhline(0, color="k", lw=0.4)
                for y in (act_lim, -act_lim):
                    ax.axhline(y, color="r", lw=0.5, ls=":")
                ax.set_ylim(-1, 1)
                ax.set_ylabel("acción")

                for a in axs[i]:
                    a.grid(alpha=0.25)

            axs[0, 0].set_title(f"Potencia generada (P azul, Q naranja)-día {dia}")
            axs[0, 1].set_title(f"Acciones-día {dia}")
            axs[0, 1].legend(loc="upper left", fontsize=7)
            for a in axs[-1]:
                a.set_xlabel("hora del día")
                a.set_xticks(range(0, 25, 4))
            fig.tight_layout()
            return fig


        fig = plot_agents(test_archive)
        plt.show()
        # fig.savefig("agentes.png", dpi=150)


        def to_time_by_bus(bv, n_bus=33):
            parts = []
            for x in bv:
                a = np.asarray(x, dtype=float)
                if a.ndim >= 2 and a.shape[-1] != n_bus and a.shape[-2] == n_bus:
                    a = np.swapaxes(a, -1, -2)  # estaba como (33, T)

                parts.append(a.reshape(-1, n_bus))  # (33,) -> (1,33) ; (T,33) -> (T,33)

            return np.concatenate(parts, axis=0)

        ######## Gráfico de Tensión por nodo ########

        v = to_time_by_bus(test_archive["bus_voltage"])
        v_mean = np.nanmean(v, axis=0)  # un valor por nodo
        v_min = np.nanmin(v, axis=0)
        v_max = np.nanmax(v, axis=0)
        print("nodo con tensión mínima: ", v.min(0).argmin() + 1)
        print("nodo con tensión máxima: ", v.max(0).argmax() + 1)

        num_buses = len(v_mean)
        buses = [str(i) for i in range(1, num_buses + 1)]
        angles = np.linspace(0, 2 * np.pi, num_buses, endpoint=False).tolist()
        # Cerrar los polígonos uniendo el último nodo con el primero
        angles += angles[:1]
        avg_v_verano = np.append(v_mean, v_mean[0])
        min_v_verano = np.append(v_min, v_min[0])
        max_v_verano = np.append(v_max, v_max[0])
        # avg_v_invierno = np.append(avg_v_invierno, avg_v_invierno[0])
        # Generar la gráfica polar comparativa
        fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig1.suptitle(f'{algoritmo}-{clase}')
        ax1.set_theta_offset(np.pi / 2)
        ax1.set_theta_direction(-1)
        # Configurar límites radiales
        r_min = min(min_v_verano.min(), min_v_verano.min()) - 0.02
        r_max = max(max_v_verano.max(), max_v_verano.max()) + 0.02
        ax1.set_rlim(r_min, r_max)
        # --- PINTAR BANDAS DE VOLTAJE ---
        theta_contour = np.linspace(0, 2 * np.pi, 500)
        # Zona normal (0.95 a 1.05 pu) en verde
        ax1.fill_between(
            theta_contour, 0.95, 1.05, color="green", alpha=0.15, zorder=0
        )
        # Zona fuera de rango (< 0.95 y > 1.05 pu) en rojo
        ax1.fill_between(theta_contour, r_min, 0.95, color="red", alpha=0.10, zorder=0)
        ax1.fill_between(theta_contour, 1.05, r_max, color="red", alpha=0.10, zorder=0)
        # Trazar las series de cada estación
        ax1.plot(
            angles,
            avg_v_verano,
            label="AVG - Verano",
            color="blue",
            linewidth=1.8,
            zorder=2,
        )
        ax1.plot(
            angles,
            min_v_verano,
            label="MIN - Verano",
            color="green",
            linewidth=1.8,
            zorder=2,
        )
        ax1.plot(
            angles,
            max_v_verano,
            label="MAX - Verano",
            color="red",
            linewidth=1.8,
            zorder=2,
        )
        # Configurar etiquetas del eje y formato
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(buses, fontsize=8)
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax1.legend(loc="lower right", bbox_to_anchor=(1.25, 0.1))
        plt.title(f"Perfil Promedio de Tensión por Nodo-día {dia}", pad=20)
        plt.tight_layout()
        fig1.show()

        ######## Gráfico de Pot Reactica por nodo ########
        q = to_time_by_bus(test_archive["bus_reactive"])
        q = q[:, 1:]
        q_mean = np.nanmean(q, axis=0)  # un valor por nodo
        q_min = np.nanmin(q, axis=0)
        q_max = np.nanmax(q, axis=0)
        print("nodo con Q mínima: ", q.min(0).argmin() + 1)
        print("nodo con Q máxima: ", q.max(0).argmax() + 1)

        num_buses = len(q_mean)
        buses = [str(i) for i in range(1, num_buses + 1)]
        angles = np.linspace(0, 2 * np.pi, num_buses, endpoint=False).tolist()
        # Cerrar los polígonos uniendo el último nodo con el primero
        angles += angles[:1]
        avg_q_verano = np.append(q_mean, q_mean[0])
        min_q_verano = np.append(q_min, q_min[0])
        max_q_verano = np.append(q_max, q_max[0])
        # avg_v_invierno = np.append(avg_v_invierno, avg_v_invierno[0])
        # Generar la gráfica polar comparativa
        fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig2.suptitle(f'{algoritmo}-{clase}')
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        # Configurar límites radiales
        r_min = min(min_q_verano.min(), min_q_verano.min()) - 0.02
        r_max = max(max_q_verano.max(), max_q_verano.max()) + 0.02
        ax2.set_rlim(r_min, r_max)
        # Trazar las series de cada estación
        ax2.plot(
            angles,
            avg_q_verano,
            label="AVG - Verano",
            color="blue",
            linewidth=1.8,
            zorder=2,
        )
        ax2.plot(
            angles,
            min_q_verano,
            label="MIN - Verano",
            color="green",
            linewidth=1.8,
            zorder=2,
        )
        ax2.plot(
            angles,
            max_q_verano,
            label="MAX - Verano",
            color="red",
            linewidth=1.8,
            zorder=2,
        )
        # Configurar etiquetas del eje y formato
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(buses, fontsize=8)
        # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax2.legend(loc="lower right", bbox_to_anchor=(1.25, 0.1))
        plt.title(f"Perfil Promedio de Potencia Reactiva por Nodo-día {dia}", pad=20)
        plt.tight_layout()
        fig2.show()