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


    # Primeros y ultimos datos
    print("Primeros y ultimos datos:\n")
    print(txt.head(-40))
    print("\n%" + "="*79 + "\n")


    # Muestra de cinco datos en LaTeX
    print("Muestra aleatoria de 5 datos:\n")
    print(txt.sample(5))
    print("\n%" + "="*79 + "\n")
    

    # Separar metadatos y datos
    meta_data = txt.head(9)
    data = txt.iloc[9:].reset_index(drop=True)
    meta_data = meta_data.transpose()   # Transponer los metadatos

    # Renombrar las columnas de los metadatos usando la primera fila
    meta_data.columns = meta_data.iloc[0]
    meta_data = meta_data[1:].reset_index()

    # Para los datos, la primera columna es 'Year'
    data = data.rename(columns={data.columns[0]: 'Year'})


    # Mostramos los metadatos
    print("Metadatos:\n")
    print(meta_data.to_latex( # Quitar indices al transponer
        index=False,
        caption=r"Metadatos",
        label=r"tab:metadatos"
    ))
    print("\n%" + "="*79 + "\n")


    ## Data
    # Estadisticas basicas en LaTeX
    print("Estadisticas basicas (LaTeX):")
    print((data.describe(include="all").T).to_latex( # Transponer para espacio
        float_format="%.3f",   # Formato de números decimales
        caption=r"Estadísticas descriptivas de los datos",
        label=r"tab:estadisticas_basicas"
    ))
    print("\n%" + "="*79 + "\n")


    info = pd.DataFrame({
        'Columna': data.columns,
        'Tipo': data.dtypes.values,
        'No nulos': data.count().values,
        'Valores faltantes': data.isna().sum().values,
        'Porcentaje faltante': data.isna().sum() / len(data) * 100
    })
    
    print("Informacion de los datos:\n")
    print(info.to_latex(
        index=False,
        caption=r"Tipos de datos, valores no nulos y faltantes por columna",
        label=r"tab:info_datos"
    ))


    # ver graficamente como se distribuyen los faltantes
    plt.figure(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(data.isna(), cbar=False, cmap="gray")
    plt.xticks(rotation=45) ; plt.suptitle("Datos faltantes")
    plt.savefig("./reports/figures/faltantes.png")  # guardar figura
    plt.close()
