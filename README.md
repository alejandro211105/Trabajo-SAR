# SAR - Motor de Búsqueda de Wikipedia

Motor de búsqueda de artículos de Wikipedia implementado en Python, con índice invertido, búsqueda booleana (AND/NOT), búsqueda posicional por frases y búsqueda semántica mediante embeddings y KDTree.

Proyecto universitario de la asignatura **Sistemas de Almacenamiento y Recuperación de Información (SAR)** de la UPV.

---

## Archivos en este repositorio

| Archivo | Descripción |
|---|---|
| `SAR_lib.py` | Implementación principal de la clase `SAR_Indexer` |
| `SAR_lib_comentada.py` | Misma implementación con comentarios detallados |

---

## Archivos necesarios para ejecutar el proyecto

Los siguientes archivos forman parte del proyecto pero **no están incluidos en este repositorio** ya que son proporcionados por la asignatura:

| Archivo | Descripción |
|---|---|
| `SAR_Indexer.py` | Programa principal de indexación. No se modifica. |
| `SAR_Searcher.py` | Programa principal de búsqueda. No se modifica. |
| `SAR_semantics.py` | Librería para embeddings y KDTree. No se modifica. |
| `SAR_semantics_demo.ipynb` | Notebook de demostración de la librería semántica. |
| `datos/` | Directorio con los artículos de Wikipedia en formato `.json` |

---

## Uso

### Indexación básica
```bash
python SAR_Indexer.py datos/ indice.bin
```

### Indexación con búsqueda posicional
```bash
python SAR_Indexer.py datos/ indice_pos.bin -P
```

### Indexación con búsqueda semántica
```bash
python SAR_Indexer.py datos/ indice_sem.bin -P -S
```

### Búsqueda interactiva
```bash
python SAR_Searcher.py indice.bin
```

### Búsqueda con una consulta directa
```bash
python SAR_Searcher.py indice.bin -Q "españa madrid"
```

### Búsqueda semántica
```bash
python SAR_Searcher.py indice_sem.bin -S 0.5 -Q "inteligencia artificial"
```

### Reranking semántico
```bash
python SAR_Searcher.py indice_sem.bin -R -Q "machine learning"
```

---

## Funcionalidades implementadas

- Indexación de artículos desde ficheros `.json`
- Búsqueda booleana con AND implícito y NOT
- Búsqueda posicional por frases entre comillas
- Búsqueda semántica mediante embeddings y KDTree
- Reranking semántico de resultados booleanos

---

## Dependencias

```bash
pip install nltk spacy sentence-transformers transformers torch
```
