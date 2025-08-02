import numpy as np
from wordfreq import top_n_list
import pyphen

def load_german_words(n=500_000):
    # Lade die ersten n deutschen Wörter
    words = top_n_list('de', n=n)[:n]
    # Silbentrennung
    dic = pyphen.Pyphen(lang='de')
    syllables = [dic.inserted(w).split('-') for w in words]
    syllable_counts = [len(s) for s in syllables]
    # Array (n×3): Wort, Silbenliste, Silbenanzahl
    arr = np.empty((n, 3), dtype=object)
    arr[:,0] = words
    arr[:,1] = syllables
    arr[:,2] = syllable_counts
    return arr

# Initiales Array erstellen
arr = load_german_words()

def add_coloum(func):
    """
    Fügt dem globalen Array arr eine neue Spalte hinzu.
    func: Funktion, die auf jedes Wort (arr[:,0]) angewendet wird.
    """
    global arr
    new_col = [func(w) for w in arr[:,0]]
    arr = np.column_stack((arr, new_col))
    return arr

# Beispiel: Neue Spalte mit Großschreibung
# arr = add_coloum(lambda w: w.upper())
print(arr.shape)  # (500000, 4)