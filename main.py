import pandas as pd
from metodos.tiempo_de_encendido import promedio_bomba_encendida
from metodos.otrosMetodos import promedio_caudal
from metodos.GraphC import graficar_caudal_y_detectar_intervalos
from metodos.ajustar import ajustar_hd_k_A_desde_df

#Lector de archivos. CVSpath deberia cambiar para que del front end se pueda decidir que archivo leer
csv_path  = "pruebas/Agrícola Río Jara - Camarico_49059220-9c85-11ef-b067-1b5a3182c1f7.csv"
df = pd.read_csv(csv_path)

# grafica el caudal en el tiempo tomando un dataset. Devuelve un dataset con las timestamps en las que la bomba esta encendida.
df["ts"] = pd.to_datetime(df["ts"])
intervalos = graficar_caudal_y_detectar_intervalos(df)
print(intervalos)

##Toma el promedio del caudal y lo imprime
promedio_caudal(df, intervalos)

##en pgrogreso
##graficar_caudal_suavizado(df)


##calcula C, H_s, H_d, K, A y tau
res = ajustar_hd_k_A_desde_df(df, ciclo=1, col_timestamp="ts", col_caudal="caudal_ls", col_nivel="nivel_m")
print(res)