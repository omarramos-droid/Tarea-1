import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import anderson
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import seaborn as sns

# =====================
# 1. CARGA Y PREPARACIÓN DE DATOS
# =====================


def cargar_datos_con_rangos(ruta_archivo):
    ruta_archivo = Path(ruta_archivo)

    #detect file extension
    if "txt" in ruta_archivo.suffix:
        txt = txt = pd.read_csv(ruta_archivo, encoding='ANSI', sep=None, header=3, engine="python")
    elif "xlsx" in ruta_archivo.suffix:
        txt = pd.read_excel(ruta_archivo, header=3)
    else:
        txt= pd.DataFrame()
        

    meta_data = txt.head(9)
    data = txt.iloc[9:].reset_index(drop=True)

    data = data.rename(columns={data.columns[0]: 'Year'})

    first_years, last_years = {}, {}
    for _, row in meta_data.iterrows():
        if 'First year' in str(row.iloc[0]):
            for j, col in enumerate(data.columns[1:], 1):
                first_years[col] = row.iloc[j]
        elif 'Last year' in str(row.iloc[0]):
            for j, col in enumerate(data.columns[1:], 1):
                last_years[col] = row.iloc[j]

    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    return data, first_years, last_years, meta_data

def ajustar_rangos_temporales(data, first_years, last_years):
    data_ajustado = data.copy()
    for col in data.columns[1:]:
        mask = (data['Year'] < first_years[col]) | (data['Year'] > last_years[col])
        data_ajustado.loc[mask, col] = np.nan
    return data_ajustado

# =====================
# 2. ANÁLISIS DE FALTANTES POR RANGO
# =====================
def contar_faltantes_por_rango(data, first_years, last_years):
    resultados = []
    for col in data.columns:
        if col == 'Year':
            continue
        mask = (data['Year'] >= first_years[col]) & (data['Year'] <= last_years[col])
        datos_en_rango = data.loc[mask, col]
        faltantes = datos_en_rango.isna().sum()
        total = len(datos_en_rango)
        resultados.append({
            'Columna': col,
            'First Year': first_years[col],
            'Last Year': last_years[col],
            'Total en rango válido': total,
            'Faltantes en rango válido': faltantes,
            '% Faltantes en rango': round((faltantes / total * 100), 2)
        })
    return pd.DataFrame(resultados)

# =====================
# 3. REGRESIÓN LINEAL SIMPLE
# =====================
def regresion_lineal_por_columna(data, first_years, last_years):
    resultados = []
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if col == 'Year':
            continue
        
        mask = (data['Year'] >= first_years[col]) & (data['Year'] <= last_years[col])
        datos_filtrados = data.loc[mask, ['Year', col]].dropna()
        
        X = datos_filtrados['Year'].values
        y = datos_filtrados[col].values
        
        X_matrix = np.column_stack([np.ones(len(X)), X])
        coefficients, _, _, _ = lstsq(X_matrix, y)
        b0, b1 = coefficients
        
        y_pred = b0 + b1 * X
        residuals = y - y_pred
        
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum(residuals**2)
        r_squared = 1 - (ss_residual / ss_total)
        
        ss_reg = np.sum((y_pred - np.mean(y))**2)
        ss_res = np.sum(residuals**2)
        df_reg, df_res = 1, len(y) - 2
        F_stat = (ss_reg / df_reg) / (ss_res / df_res)
        p_value = 1 - stats.f.cdf(F_stat, df_reg, df_res)
        
        resultados.append({
            'Columna': col,
            'N_datos': len(datos_filtrados),
            'Intercepto': b0,
            'Pendiente': b1,
            'R_cuadrado': r_squared,
            'P_valor': p_value,
            'Regresion_valida': 'Sí' if p_value < 0.05 else 'No'
        })
    
    return pd.DataFrame(resultados)

# =====================
# 4. TEST DE NORMALIDAD DE RESIDUOS
# =====================
def test_normalidad_anderson_residuos(data, first_years, last_years, alpha=0.05):
    resultados = []
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if col == 'Year':
            continue
            
        mask = (data['Year'] >= first_years[col]) & (data['Year'] <= last_years[col])
        datos_filtrados = data.loc[mask, ['Year', col]].dropna()
            
        X = datos_filtrados['Year'].values
        y = datos_filtrados[col].values
        
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        
        residuos = model.resid
        
        result = anderson(residuos)
        estadistico = result.statistic
        
        p_valor_aproximado = None
        for i, nivel in enumerate(result.significance_level):
            if estadistico < result.critical_values[i]:
                p_valor_aproximado = result.significance_level[i] / 100
                break
        
        if p_valor_aproximado is None:
            p_valor_aproximado = 0.001
            
        normalidad = 'Sí' if p_valor_aproximado > alpha else 'No'
        
        resultados.append({
            'Columna': col,
            'N_datos': len(datos_filtrados),
            'Estadistico_AD': estadistico,
            'p_valor_aproximado': p_valor_aproximado,
            'Normalidad_razonable': normalidad,
            'R_cuadrado': model.rsquared,
            'Pendiente': model.params[1],
            'P_valor_regresion': model.f_pvalue
        })
    
    return pd.DataFrame(resultados)

# =====================
# 5. DETECCIÓN DE OUTLIERS EN RESIDUOS
# =====================
def detectar_outliers_hibrido_residuos(data, first_years, last_years, normalidad_df, output_dir='outputs/outliers'):
    os.makedirs(output_dir, exist_ok=True)
    
    resultados = []
    figuras_generadas = []
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if col == 'Year':
            continue
            
        normalidad_info = normalidad_df[normalidad_df['Columna'] == col]
        es_normal = normalidad_info['Normalidad_razonable'].values[0]
        
        mask = (data['Year'] >= first_years[col]) & (data['Year'] <= last_years[col])
        datos_filtrados = data.loc[mask, ['Year', col]].dropna()
            
        X = datos_filtrados['Year'].values
        y = datos_filtrados[col].values
        
        # Verificar si se puede hacer regresión
        puede_regresion = len(datos_filtrados) >= 2
        
        if puede_regresion:
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit()
            residuos = model.resid
            valores_ajustados = model.fittedvalues
            
            if es_normal == 'Sí':
                influence = model.get_influence()
                cook_distance = influence.cooks_distance[0]
                n = len(y)
                umbral_cook = 4/n
                outliers_mask = cook_distance > umbral_cook
                metodo = 'cook_distance'
            else:
                Q1 = np.percentile(residuos, 25)
                Q3 = np.percentile(residuos, 75)
                IQR = Q3 - Q1
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                metodo = 'IQR_residuos'
                outliers_mask = (residuos < limite_inferior) | (residuos > limite_superior)
        else:
            # No se puede hacer regresión, usar método simple para outliers
            metodo = 'sin_regresion'
            Q1 = np.percentile(y, 25)
            Q3 = np.percentile(y, 75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            outliers_mask = (y < limite_inferior) | (y > limite_superior)
            residuos = y - np.mean(y)
        
        outliers_count = np.sum(outliers_mask)
        outliers_years = datos_filtrados['Year'].values[outliers_mask] if puede_regresion else X[outliers_mask]
        outliers_values = y[outliers_mask]
        
        resultado = {
            'Columna': col,
            'N_datos': len(y),
            'Metodo': metodo,
            'Outliers_percent': (outliers_count / len(y)) * 100
        }
        
        if metodo == 'cook_distance':
            resultado['Limite_cook'] = umbral_cook
            resultado['Cook_distance_max'] = np.max(cook_distance)
        elif metodo == 'IQR_residuos':
            resultado['Limite_inferior_residuos'] = limite_inferior
            resultado['Limite_superior_residuos'] = limite_superior
        else:
            resultado['Limite_inferior'] = limite_inferior
            resultado['Limite_superior'] = limite_superior
        
        resultados.append(resultado)
        
        # Crear figura según si se pudo hacer regresión o no
        if puede_regresion:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            if metodo == 'cook_distance':
                ax1.scatter(range(len(cook_distance)), cook_distance, alpha=0.7, color='blue')
                ax1.axhline(y=umbral_cook, color='red', linestyle='--', label=f'Umbral: {umbral_cook:.3f}')
                
                if outliers_count > 0:
                    outlier_indices = np.where(outliers_mask)[0]
                    ax1.scatter(outlier_indices, cook_distance[outliers_mask], color='red', s=100, edgecolors='black')
                
                ax1.set_xlabel('Índice de Observación')
                ax1.set_ylabel('Distancia de Cook')
                ax1.set_title(f'Distancia de Cook - {col}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.scatter(valores_ajustados[~outliers_mask], residuos[~outliers_mask], alpha=0.7, color='blue')
                
                if outliers_count > 0:
                    ax1.scatter(valores_ajustados[outliers_mask], residuos[outliers_mask], color='red', s=100, edgecolors='black', marker='X')
                
                ax1.axhline(y=limite_superior, color='orange', linestyle='--', label=f'Límite superior: {limite_superior:.3f}')
                ax1.axhline(y=limite_inferior, color='orange', linestyle='--', label=f'Límite inferior: {limite_inferior:.3f}')
                ax1.axhline(y=0, color='green', linestyle='-', alpha=0.5)
                
                ax1.set_xlabel('Valores Ajustados')
                ax1.set_ylabel('Residuos')
                ax1.set_title(f'Residuos vs Ajustados - {col}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Segunda gráfica con regresión
            ax2.scatter(X[~outliers_mask], y[~outliers_mask], alpha=0.7, color='blue')
            
            if outliers_count > 0:
                if metodo == 'cook_distance':
                    ax2.scatter(outliers_years, outliers_values, color='red', s=100, edgecolors='black')
                else:
                    ax2.scatter(outliers_years, outliers_values, color='red', s=100, edgecolors='black', marker='X')
            
            # Graficar línea de regresión
            x_line = np.linspace(X.min(), X.max(), 100)
            y_line = model.params[0] + model.params[1] * x_line
            ax2.plot(x_line, y_line, color='green', linestyle='-', label=f'Regresión: y = {model.params[0]:.3f} + {model.params[1]:.3f}x')
            
            ax2.set_xlabel('Año')
            ax2.set_ylabel('Valor')
            ax2.set_title('Serie Temporal con Regresión')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        else:
            # Solo una gráfica cuando no hay regresión
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            ax.scatter(X[~outliers_mask], y[~outliers_mask], alpha=0.7, color='blue', label='Datos normales')
            
            if outliers_count > 0:
                ax.scatter(outliers_years, outliers_values, color='red', s=100, edgecolors='black', marker='X', label='Outliers')
            
            ax.axhline(y=limite_superior, color='orange', linestyle='--', label=f'Límite superior: {limite_superior:.3f}')
            ax.axhline(y=limite_inferior, color='orange', linestyle='--', label=f'Límite inferior: {limite_inferior:.3f}')
            
            ax.set_xlabel('Año')
            ax.set_ylabel('Valor')
            ax.set_title(f'Serie Temporal - {col}\n(Método: IQR directo - Sin regresión)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{col} - Método: {metodo} - Outliers: {outliers_count}', fontsize=14, fontweight='bold')
        
        filename = f"{col.replace('/', '_').replace(' ', '_')}_outliers.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        figuras_generadas.append(filepath)
    
    resultados_df = pd.DataFrame(resultados)
    resultados_path = os.path.join(output_dir, 'tabla_outliers_residuos.csv')
    resultados_df.to_csv(resultados_path, index=False, encoding='utf-8')
    
    return resultados_df, figuras_generadas

# =====================
# 6. FUNCIÓN PRINCIPAL COMPLETA
# =====================
def analisis_completo_outliers(ruta_archivo, output_base_dir='outputs'):
    os.makedirs(output_base_dir, exist_ok=True)
    
    data, first_years, last_years, meta_data = cargar_datos_con_rangos(ruta_archivo)
    data_ajustado = ajustar_rangos_temporales(data, first_years, last_years)
    
    normalidad_df = test_normalidad_anderson_residuos(data_ajustado, first_years, last_years)
    normalidad_path = os.path.join(output_base_dir, 'test_normalidad_residuos.csv')
    normalidad_df.to_csv(normalidad_path, index=False, encoding='utf-8')
    
    outliers_dir = os.path.join(output_base_dir, 'outliers_residuos')
    outliers_df, figuras = detectar_outliers_hibrido_residuos(data_ajustado, first_years, last_years, normalidad_df, outliers_dir)
    
    return {
        'data': data_ajustado,
        'normalidad': normalidad_df,
        'outliers': outliers_df,
        'figuras': figuras
    }

def imputacion_diferenciada_normalidad(data, first_years, last_years, normalidad_df, outliers_df=None):
    """
    Imputación diferenciada según normalidad de los datos
    Para normales: regresión + incertidumbre
    Para no normales: mediana + ruido proporcional al IQR
    """
    data_imputado = data.copy()
    estrategias_imputacion = {}
    
    for col in data.select_dtypes(include=[np.number]).columns:
        if col == 'Year':
            continue
            
        normalidad_info = normalidad_df[normalidad_df['Columna'] == col]
        if len(normalidad_info) == 0:
            continue
            
        es_normal = normalidad_info['Normalidad_razonable'].values[0]
        mask_rango = (data['Year'] >= first_years[col]) & (data['Year'] <= last_years[col])
        
        datos_rango = data.loc[mask_rango, ['Year', col]]
        datos_validos = datos_rango.dropna()
        
        if len(datos_validos) < 2:
            media = datos_validos[col].mean() if len(datos_validos) > 0 else 0
            data_imputado.loc[mask_rango & data[col].isna(), col] = media
            estrategias_imputacion[col] = 'media_simple'
            continue
        
        if es_normal == 'Sí':
            try:
                X = datos_validos['Year'].values.astype(float)
                y = datos_validos[col].values
                
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit()
                
                beta0, beta1 = model.params
                residuos = model.resid
                n = len(datos_validos)
                x_mean = X.mean()
                
                mask_faltantes = mask_rango & data[col].isna()
                fechas_faltantes = data.loc[mask_faltantes, 'Year'].values.astype(float)
                
                if len(fechas_faltantes) > 0:
                    y_pred = beta0 + beta1 * fechas_faltantes
                    
                    SSE = np.sum(residuos**2)
                    MSRes = SSE / (n - 2)
                    Sxx = np.sum((X - x_mean)**2)
                    
                    Var_pred = MSRes * (1 + 1/n + (fechas_faltantes - x_mean)**2 / Sxx)
                    SE_pred = np.sqrt(Var_pred)
                    
                    t_values = stats.t.rvs(df=n-2, size=len(fechas_faltantes))
                    y_imputado = y_pred + SE_pred * t_values
                    
                    for i, year in enumerate(fechas_faltantes):
                        idx = data_imputado[(data_imputado['Year'] == year) & mask_rango].index
                        if len(idx) > 0:
                            data_imputado.loc[idx, col] = y_imputado[i]
                    
                    estrategias_imputacion[col] = 'regresion_incertidumbre'
                
            except Exception:
                media = datos_validos[col].mean()
                data_imputado.loc[mask_rango & data[col].isna(), col] = media
                estrategias_imputacion[col] = 'media_fallback'
                
        else:
            mask_faltantes = mask_rango & data[col].isna()
            años_faltantes = data.loc[mask_faltantes, 'Year'].values
            
            if len(años_faltantes) > 0:
                mediana = datos_validos[col].median()
                Q1 = datos_validos[col].quantile(0.25)
                Q3 = datos_validos[col].quantile(0.75)
                IQR_val = Q3 - Q1
                
                factor_ruido = 0.25
                escala_ruido = factor_ruido * IQR_val
                
                np.random.seed(42)
                ruido = np.random.normal(loc=0, scale=escala_ruido, size=len(años_faltantes))
                
                valores_imputados = mediana + ruido
                
                for i, año in enumerate(años_faltantes):
                    idx = data_imputado[(data_imputado['Year'] == año) & mask_rango].index
                    if len(idx) > 0:
                        data_imputado.loc[idx, col] = valores_imputados[i]
                    else:
                        idx = data_imputado[data_imputado['Year'] == año].index
                        if len(idx) > 0:
                            data_imputado.loc[idx, col] = valores_imputados[i]
                
                estrategias_imputacion[col] = 'mediana_ruido_iqr'
    
    return data_imputado, estrategias_imputacion

def graficar_imputacion_resultados(data_original, data_imputado, estrategias_imputacion, 
                                  first_years, last_years, normalidad_df, output_dir='outputs/imputacion'):
    """
    Graficar resultados de la imputación diferenciada en el mismo formato que outliers
    """
    os.makedirs(output_dir, exist_ok=True)
    figuras_generadas = []
    
    for col in data_original.select_dtypes(include=[np.number]).columns:
        if col == 'Year' or col not in estrategias_imputacion:
            continue
            
        metodo = estrategias_imputacion[col]
        mask_rango = (data_original['Year'] >= first_years[col]) & (data_original['Year'] <= last_years[col])
        
        original_rango = data_original.loc[mask_rango, ['Year', col]]
        imputado_rango = data_imputado.loc[mask_rango, ['Year', col]]
        
        mask_imputados = original_rango[col].isna() & imputado_rango[col].notna()
        puntos_imputados = imputado_rango[mask_imputados]
        
        datos_originales = original_rango.dropna()
        
        if len(datos_originales) < 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            ax.scatter(datos_originales['Year'], datos_originales[col], 
                      alpha=0.7, color='blue', s=60, label='Datos originales')
            
            if len(puntos_imputados) > 0:
                ax.scatter(puntos_imputados['Year'], puntos_imputados[col], 
                          color='green', s=100, edgecolors='black', marker='X',
                          label='Datos imputados', zorder=5)
            
            ax.set_xlabel('Año')
            ax.set_ylabel('δ13C (‰)')
            ax.set_title(f'Sitio {col} - Imputación: {metodo}\nDatos imputados: {len(puntos_imputados)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            if metodo == 'mediana_ruido_iqr':
                mediana = datos_originales[col].median()
                
                if len(puntos_imputados) > 0:
                    ax1.hist(puntos_imputados[col], bins=15, alpha=0.7, color='green', 
                            label='Datos imputados')
                    ax1.axvline(x=mediana, color='red', linestyle='--', linewidth=2, 
                               label=f'Mediana: {mediana:.3f}')
                    ax1.set_xlabel('δ13C (‰)')
                    ax1.set_ylabel('Frecuencia')
                    ax1.set_title('Distribución de Datos Imputados')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                else:
                    ax1.text(0.5, 0.5, 'No hay datos imputados', 
                            ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Distribución de Datos Imputados')
                
                ax2.scatter(datos_originales['Year'], datos_originales[col], 
                           alpha=0.7, color='blue', s=60, label='Datos originales')
                
                if len(puntos_imputados) > 0:
                    ax2.scatter(puntos_imputados['Year'], puntos_imputados[col], 
                               color='green', s=100, edgecolors='black', marker='X',
                               label='Datos imputados', zorder=5)
                
                x_min = datos_originales['Year'].min()
                x_max = datos_originales['Year'].max()
                ax2.axhline(y=mediana, color='red', linestyle='--', linewidth=2,
                           label=f'Mediana: {mediana:.3f}')
                
                ax2.set_xlabel('Año')
                ax2.set_ylabel('δ13C (‰)')
                ax2.set_title('Serie Temporal con Mediana')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
            else:
                X = datos_originales['Year'].values
                y = datos_originales[col].values
                
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit()
                
                residuos = model.resid
                valores_ajustados = model.fittedvalues
                
                normalidad_info = normalidad_df[normalidad_df['Columna'] == col]
                es_normal = normalidad_info['Normalidad_razonable'].values[0] if len(normalidad_info) > 0 else 'No'
                
                if es_normal == 'Sí':
                    influence = model.get_influence()
                    cook_distance = influence.cooks_distance[0]
                    n = len(y)
                    umbral_cook = 4/n
                    
                    ax1.scatter(range(len(cook_distance)), cook_distance, alpha=0.7, color='blue')
                    ax1.axhline(y=umbral_cook, color='red', linestyle='--', label=f'Umbral: {umbral_cook:.3f}')
                    ax1.set_xlabel('Índice de Observación')
                    ax1.set_ylabel('Distancia de Cook')
                    ax1.set_title(f'Distancia de Cook - {col}')
                    
                else:
                    Q1 = np.percentile(residuos, 25)
                    Q3 = np.percentile(residuos, 75)
                    IQR = Q3 - Q1
                    limite_inferior = Q1 - 1.5 * IQR
                    limite_superior = Q3 + 1.5 * IQR
                    
                    ax1.scatter(valores_ajustados, residuos, alpha=0.7, color='blue')
                    ax1.axhline(y=limite_superior, color='orange', linestyle='--', label=f'Límite superior: {limite_superior:.3f}')
                    ax1.axhline(y=limite_inferior, color='orange', linestyle='--', label=f'Límite inferior: {limite_inferior:.3f}')
                    ax1.axhline(y=0, color='green', linestyle='-', alpha=0.5)
                    ax1.set_xlabel('Valores Ajustados')
                    ax1.set_ylabel('Residuos')
                    ax1.set_title(f'Residuos vs Ajustados - {col}')
                
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.scatter(datos_originales['Year'], datos_originales[col], 
                           alpha=0.7, color='blue', s=60, label='Datos originales')
                
                if len(puntos_imputados) > 0:
                    ax2.scatter(puntos_imputados['Year'], puntos_imputados[col], 
                               color='green', s=100, edgecolors='black', marker='X',
                               label='Datos imputados', zorder=5)
                
                x_line = np.linspace(X.min(), X.max(), 100)
                y_line = model.params[0] + model.params[1] * x_line
                ax2.plot(x_line, y_line, color='red', linestyle='-', 
                        label=f'Regresión: y = {model.params[0]:.3f} + {model.params[1]:.3f}x')
                
                ax2.set_xlabel('Año')
                ax2.set_ylabel('δ13C (‰)')
                ax2.set_title('Serie Temporal con Regresión')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{col} - Método: {metodo} - Datos imputados: {len(puntos_imputados)}', 
                    fontsize=14, fontweight='bold')
        
        filename = f"{col.replace('/', '_').replace(' ', '_')}_imputacion.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        figuras_generadas.append(filepath)
    
    return figuras_generadas

# Función principal de imputación
def proceso_imputacion_completo(data, first_years, last_years, normalidad_df, outliers_df=None):
    """
    Proceso completo de imputación diferenciada
    """
    data_imputado, estrategias = imputacion_diferenciada_normalidad(
        data, first_years, last_years, normalidad_df, outliers_df
    )
    
    figuras = graficar_imputacion_resultados(
        data, data_imputado, estrategias, first_years, last_years, normalidad_df
    )
    
    resumen_imputacion = pd.DataFrame.from_dict(estrategias, orient='index', 
                                               columns=['Metodo_imputacion'])
    
    return data_imputado, estrategias, figuras

def estimar_densidad_graficas(df_isotopes, df, columnnumber):
    """
    Estima y grafica la densidad de datos isotópicos para una columna específica
    
    Parameters:
    df_isotopes: DataFrame con los datos isotópicos
    df: DataFrame original con metadatos
    columnnumber: número de columna a analizar (0 al 24)
    """
    columncode = df_isotopes.columns[columnnumber]
    sitename = df.iloc[1][columnnumber + 1]
    
    # Obtener datos sin valores nulos
    datos = df_isotopes[columncode].dropna()
    
    if len(datos) == 0:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Histograma
    sns.histplot(datos, kde=False, ax=axes[0], color="pink")
    axes[0].set_title(f"Histograma de {sitename}")
    axes[0].set_xlabel("δ¹³C (‰, VPDB)")
    
    # Densidad kernel
    sns.kdeplot(datos, ax=axes[1], color="green", fill=True)
    axes[1].set_title(f"Densidad kernel estimada de {sitename}")
    axes[1].set_xlabel("δ¹³C (‰, VPDB)")

    plt.tight_layout()
    plt.show()

    # Histograma con densidad estimada combinada
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.histplot(datos, kde=True, ax=ax, color="pink", stat="density")
    ax.set_title(f"Histograma y densidad kernel de {sitename}")
    ax.set_xlabel("δ¹³C (‰, VPDB)")
    
    plt.tight_layout()
    plt.show()

def graficar_dispersion_especies(df):
    """
    Grafica dispersión para mismas especies en diferentes sitios
    
    Parameters:
    df: DataFrame con los datos
    """
    different_species = df.iloc[:, 1:].iloc[5].value_counts().to_dict()

    for species in different_species:
        subdata_columns = df.columns[df.iloc[5] == species]
        subdata = df.iloc[10:, subdata_columns]
        subdata = subdata.set_axis(df.iloc[0, subdata_columns], axis=1)
        subdata = subdata.set_index(df.iloc[10:, 0])

        plt.figure(figsize=(8, 5))
        for location in subdata.columns:
            # Extraer datos (quitando NA)
            data_withoutna = subdata[location].dropna()
            X = data_withoutna.index.to_numpy(dtype=float)
            y = data_withoutna.to_numpy(dtype=float)
            plt.scatter(X, y, alpha=0.6, label=location)
        
        plt.title(f"Datos de la especie {species}")
        plt.ylabel("δ¹³C (‰, VPDB)")
        plt.xlabel("Año")
        plt.legend()
        plt.tight_layout()
        plt.show()

