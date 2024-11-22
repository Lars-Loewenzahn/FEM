import nbformat

# Lade das Notebook
with open('num_ue5.ipynb', 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)

# Normalisiere das Notebook
nbformat.validate(notebook)
nbformat.write(notebook, 'num_ue5_normalized.ipynb')