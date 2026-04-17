import pickle

# Carga el indice posicional
with open('indice_pos.bin', 'rb') as f:
    info = pickle.load(f)
atrs = info[0]
data = dict(zip(atrs, info[1:]))
idx = data['index']
total = len(data['articles'])

def and_merge(p1, p2):
    r = []; i = j = 0
    while i < len(p1) and j < len(p2):
        if p1[i] == p2[j]: r.append(p1[i]); i += 1; j += 1
        elif p1[i] < p2[j]: i += 1
        else: j += 1
    return r

def rev_merge(p, total):
    r = []; j = 0
    for artid in range(total):
        while j < len(p) and p[j] < artid: j += 1
        if j >= len(p) or p[j] != artid: r.append(artid)
    return r

def get_artids(term):
    e = idx.get(term, None)
    if e is None: return []
    if isinstance(e, dict): return sorted(e.keys())
    return e

def get_phrase(terms):
    postings = [idx.get(t, {}) for t in terms]
    if any(not isinstance(p, dict) for p in postings): return []
    artids = sorted(postings[0].keys())
    for k in range(1, len(terms)):
        artids = and_merge(artids, sorted(postings[k].keys()))
    result = []
    for artid in artids:
        pos0 = postings[0][artid]
        for p in pos0:
            ok = all((p + k) in postings[k][artid] for k in range(1, len(terms)))
            if ok:
                result.append(artid)
                break
    return result

# Frases de prueba
cases = [
    ['civil', 'war'],
    ['de', 'la'],
    ['real', 'madrid'],
    ['nacido', 'en'],
    ['premio', 'nobel'],
]
for phrase in cases:
    r = get_phrase(phrase)
    label = '"' + ' '.join(phrase) + '"'
    print(f'{label}\t{len(r)}')

# Combinaciones frase + termino
r = and_merge(get_phrase(['de', 'la']), get_artids('historia'))
print(f'"de la" historia\t{len(r)}')

r = and_merge(get_phrase(['real', 'madrid']), get_artids('futbol'))
print(f'"real madrid" futbol\t{len(r)}')

# Terminos con NOT en indice posicional (deben seguir funcionando)
print(f'espana\t{len(get_artids("espana"))}')
print(f'NOT madrid (total artids={total})\t{len(rev_merge(get_artids("madrid"), total))}')
