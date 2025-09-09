import pandas as pd
from src.config import data_dir
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Leer los datos
    data_filename = data_dir / "raw" / "2023-002_ISONET-Project-Members_13C_Data.txt"
    txt = pd.read_csv(data_filename, encoding='ANSI', sep=None, header=3, engine="python")

    # Muestra de cinco datos
    print("Muestra de datos:\n", txt.sample(5), "\n\n")

    # informacion general
    print("Resumen de datos:\n", txt.info(), "\n\n")    # tipos y valores nulos
    print("Estadisticas basicas:\n", txt.describe(), "\n\n")    # Estadisticas


    # ver graficamente como se distribuyen los faltantes
    # plt.figure(figsize=(12, 9))
    sns.heatmap(txt.isna(), cbar=False, cmap="gray")
    plt.xticks(rotation=45) ; plt.suptitle("Datos faltantes")
    plt.show()