import pandas as pd
import numpy as np
from src.config import data_dir
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    """ Exploración inicial de los datos:
        - Información general
        - Muestra de cinco datos
        - Estadisticas básicas (En formato Latex)
        - Ver graficamente como se distribuyen los faltantes (figure)
        """
    np.random.seed(42)  # fijar semilla aleatoria para reproducibilidad


    # Leer los datos
    data_filename = data_dir / "raw" / "2023-002_ISONET-Project-Members_13C_Data.txt"
    txt = pd.read_csv(data_filename, encoding='ANSI', sep=None, header=3, engine="python")


    # Información general en LaTeX
    print("Informacion general:")
    print(txt.info())
    print("\n%" + "="*79 + "\n")


    # Muestra de cinco datos en LaTeX
    print("Muestra aleatoria de 5 datos:\n")
    print(txt.sample(5))
    print("\n%" + "="*79 + "\n")


    # Estadisticas básicas en LaTeX
    print("Estadisticas basicas (LaTeX):")
    print((txt.describe(include="all").T).to_latex( # Transponer para espacio
        float_format="%.3f",   # Formato de números decimales
        caption=r"Estadísticas descriptivas de los datos",
        label=r"tab:estadisticas_basicas"
    ))
    print("\n%" + "="*79 + "\n")


    info = pd.DataFrame({
        'Columna': txt.columns,
        'Tipo': txt.dtypes.values,
        'No nulos': txt.count().values,
        'Valores faltantes': txt.isna().sum().values,
        'Porcentaje faltante': txt.isna().sum() / len(txt) * 100
    })
    
    print(info.to_latex(
        index=False,
        caption=r"Tipos de datos, valores no nulos y faltantes por columna",
        label=r"tab:info_datos"
    ))


    # ver graficamente como se distribuyen los faltantes
    plt.figure(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(txt.isna(), cbar=False, cmap="gray")
    plt.xticks(rotation=45) ; plt.suptitle("Datos faltantes")
    plt.savefig("./reports/figures/faltantes.png")  # guardar figura
    plt.close()