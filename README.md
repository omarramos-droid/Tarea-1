# Tarea-1

Código e informe de la Tarea 1.

## :open_file_folder: Estructura del proyecto

- `data/` &rarr; lugar de los datasets.
- `src/` &rarr; código python a reutilizar.
- `reports/` &rarr; figuras y reporte final.
- `main.py` &rarr; script principal.

## Creación de entorno

El archivo `environment.yml` es para poder crear el entorno con conda

```bash
conda env create -f environment.yml
```
y lo activamos con 
```bash
conda activate <ENV-NAME>
```
donde `<ENV-NAME>` es el nombre del entorno que se encuentra en `environment.yml`.

Pero realmente cualquier entorno con `pandas`, `numpy`, `scipy` y `matplotlib` sirve (reutilizar entornos ayuda al planeta). 
