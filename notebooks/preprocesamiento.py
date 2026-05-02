"""
Preprocesamiento de datos: lee desde Data_cruda/ y escribe en data_preprocesada/.

Ir completando los pasos (limpieza, codificación, splits, etc.) según el dataset.
"""

from pathlib import Path

# Raíz del repositorio (padre de notebooks/)
ROOT = Path(__file__).resolve().parent.parent
DIR_CRUDA = ROOT / "Data_cruda"
DIR_PREPROCESADA = ROOT / "data_preprocesada"


def main() -> None:
    DIR_PREPROCESADA.mkdir(parents=True, exist_ok=True)
    # TODO: cargar CSV/Parquet desde DIR_CRUDA, transformar y guardar en DIR_PREPROCESADA
    raise NotImplementedError(
        "Define la lectura desde Data_cruda y la escritura en data_preprocesada."
    )


if __name__ == "__main__":
    main()
