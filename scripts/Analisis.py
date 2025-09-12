import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from Funciones import *

# =====================
# CONFIGURACIÓN
# =====================
ruta_archivo = Path(r"C:\Users\cyr\Desktop\Ciencia Datos\Tarea-1\data\raw\2023-002_ISONET-Project-Members_13C_Data.xlsx")
output_dir = "outputs_analisis_completo"

# =====================
# FUNCIONES AUXILIARES
# =====================
def escalamiento_hibido(data_imputado, normalidad_df):
    data_escalado = data_imputado.copy()
    estrategias_escalamiento = {}
    
    for col in data_imputado.columns[1:]:
        normalidad_info = normalidad_df[normalidad_df['Columna'] == col]
        es_normal = normalidad_info['Normalidad_razonable'].values[0]
        datos_columna = data_imputado[col].dropna()
        
        if es_normal == 'Sí':
            mean_val = datos_columna.mean()
            std_val = datos_columna.std()
            if std_val > 0:
                data_escalado[col] = (data_imputado[col] - mean_val) / std_val
                estrategias_escalamiento[col] = 'z_score'
            else:
                data_escalado[col] = 0
                estrategias_escalamiento[col] = 'constante'
        else:
            median_val = datos_columna.median()
            Q1 = datos_columna.quantile(0.25)
            Q3 = datos_columna.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                data_escalado[col] = (data_imputado[col] - median_val) / IQR
                estrategias_escalamiento[col] = 'robusto'
            else:
                data_escalado[col] = 0
                estrategias_escalamiento[col] = 'constante'
    
    return data_escalado, estrategias_escalamiento

def graficar_outliers_con_datos(data_ajustado, first_years, last_years, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    figuras_generadas = []
    
    for col in data_ajustado.columns[1:]:
        mask_rango = (data_ajustado['Year'] >= first_years[col]) & (data_ajustado['Year'] <= last_years[col])
        datos_rango = data_ajustado.loc[mask_rango, ['Year', col]].dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(datos_rango['Year'], datos_rango[col], 'o-', alpha=0.7, markersize=4)
        ax.set_xlabel('Año')
        ax.set_ylabel('δ13C (‰)')
        ax.set_title(f'Sitio {col} - Serie Temporal')
        ax.grid(True, alpha=0.3)
        
        filename = f"{col}_serie_temporal.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        figuras_generadas.append(filepath)
    
    return figuras_generadas

def estimar_densidad_graficas(df_isotopes, df, columnnumber):
    """
    Estima y grafica la densidad de datos isotópicos para una columna específica
    """
    columncode = df_isotopes.columns[columnnumber]
    sitename = df.iloc[1][columnnumber + 1]
    
    datos = df_isotopes[columncode].dropna()
    
    if len(datos) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.histplot(datos, kde=False, ax=axes[0], color="pink")
    axes[0].set_title(f"Histograma de {sitename}")
    axes[0].set_xlabel("δ¹³C (‰, VPDB)")
    
    sns.kdeplot(datos, ax=axes[1], color="green", fill=True)
    axes[1].set_title(f"Densidad kernel estimada de {sitename}")
    axes[1].set_xlabel("δ¹³C (‰, VPDB)")

    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.histplot(datos, kde=True, ax=ax, color="skyblue", stat="density")
    ax.set_title(f"Histograma y densidad kernel de {sitename}")
    ax.set_xlabel("δ¹³C (‰, VPDB)")
    
    plt.tight_layout()
    plt.show()

# =====================
# FUNCIÓN PRINCIPAL
# =====================
def main():
    data, first_years, last_years, meta_data = cargar_datos_con_rangos(ruta_archivo)
    data_ajustado = ajustar_rangos_temporales(data, first_years, last_years)
    
    # Crear df_isotopes a partir de data_ajustado
    df_isotopes = data_ajustado.set_index('Year')
    
    faltantes_df = contar_faltantes_por_rango(data_ajustado, first_years, last_years)
    faltantes_df.to_csv(f"{output_dir}/analisis_faltantes.csv", index=False)
    
    regresion_df = regresion_lineal_por_columna(data_ajustado, first_years, last_years)
    regresion_df.to_csv(f"{output_dir}/regresiones_lineales.csv", index=False)
    
    normalidad_df = test_normalidad_anderson_residuos(data_ajustado, first_years, last_years)
    normalidad_df.to_csv(f"{output_dir}/test_normalidad_residuos.csv", index=False)
    
    outliers_df, figuras_outliers = detectar_outliers_hibrido_residuos(
        data_ajustado, first_years, last_years, normalidad_df, 
        f"{output_dir}/outliers_residuos"
    )
    
    data_limpio = data_ajustado.copy()
    
    data_imputado, estrategias_imputacion, figuras_imputacion = proceso_imputacion_completo(
        data_limpio, first_years, last_years, normalidad_df, outliers_df
    )
    
    data_escalado, estrategias_escalamiento = escalamiento_hibido(data_imputado, normalidad_df)
    data_escalado.to_csv(f"{output_dir}/datos_escalados.csv", index=False)
    
    figuras_contexto = graficar_outliers_con_datos(
        data_ajustado, first_years, last_years,
        f"{output_dir}/outliers_contexto"
    )
    
    correlation_matrix = data_escalado.iloc[:, 1:].corr()
    correlation_matrix.to_csv(f"{output_dir}/matriz_correlacion.csv")
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f')
    plt.title('Matriz de Correlación entre Sitios ISONET - δ13C', fontsize=16)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlacion_sitios.png", dpi=300)
    plt.close()
    
    # Ejecutar análisis de densidad
    col_num = 2
    datos_columna = df_isotopes[df_isotopes.columns[col_num]].dropna()

    if len(datos_columna) > 0:
        estimar_densidad_graficas(df_isotopes, data, col_num)
    
    return {
        'data_original': data,
        'data_ajustado': data_ajustado,
        'data_imputado': data_imputado,
        'data_escalado': data_escalado,
        'faltantes': faltantes_df,
        'regresiones': regresion_df,
        'normalidad': normalidad_df,
        'outliers': outliers_df,
        'estrategias_imputacion': estrategias_imputacion,
        'estrategias_escalamiento': estrategias_escalamiento
    }

# =====================
# EJECUCIÓN PRINCIPAL
# =====================
if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    resultados = main()