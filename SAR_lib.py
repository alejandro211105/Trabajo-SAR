# versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel


## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"

def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": return SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()

        # PARA LA AMPLIACION
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)

            

    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """

        # 1 - Extraer frases del texto usando nltk.sent_tokenize
        frases = nltk.sent_tokenize(txt)

        # 2 - Guardar el índice de inicio para este artículo
        inicio = len(self.chuncks)
        self.artid_to_emb[artid] = inicio

        # Añadir cada frase a self.chuncks y su artid a self.chunck_index (listas paralelas)
        for frase in frases:
            self.chuncks.append(frase)
            self.chunck_index.append(artid)

        # Actualizar artid_to_emb con la tupla (inicio, fin_exclusivo)
        self.artid_to_emb[artid] = (inicio, len(self.chuncks))
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # self.chuncks y self.chunck_index son listas paralelas (mismo índice = misma frase).
        # Se eligió esta estructura porque el modelo semántico (KDTree) devuelve un índice
        # entero de chunck, y con una lista paralela se accede al artid en O(1) sin búsqueda.
        # Alternativa descartada: dict {chunck_idx: artid} — redundante y más lento de construir.
        # self.artid_to_emb guarda (inicio, fin) para localizar rápidamente todas las frases
        # de un artículo concreto sin recorrer chunck_index completo.
        # update_chuncks se llama desde index_file (no desde index_dir) porque en ese punto
        # ya tenemos el texto del artículo disponible; esperar a index_dir obligaría a releer
        # los ficheros o almacenar todos los textos en memoria.
        

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")
        # 1 - Construir el índice kdtree entrenando el modelo con todas las frases indexadas
        self.kdtree = self.model.fit(self.chuncks)
        print("done!")
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # create_kdtree se invoca al FINAL de index_dir, una vez procesados todos los ficheros.
        # Si se llamara al final de index_file (por cada fichero), el KDTree se reconstruiría
        # N veces siendo cada construcción O(n·log n) sobre el total de frases acumuladas,
        # lo que sería extremadamente ineficiente. Construirlo una sola vez al final garantiza
        # que todos los embeddings estén disponibles y que el coste sea O(n·log n) una sola vez.
        # model.fit() vectoriza self.chuncks y construye la estructura de búsqueda aproximada.


        
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()

        # Iniciar top_k en el máximo de embeddings configurado
        top_k = self.MAX_EMBEDDINGS

        while True:
            # 1 - Consultar el modelo semántico: devuelve lista de (distancia, índice_chunck)
            resultados = self.model.query(query, top_k=top_k)

            # 2 y 5 - Convertir índices de chunck a artids eliminando duplicados (con dict, sin set())
            artids = []
            vistos = {}
            for dist, idx in resultados:
                artid = self.chunck_index[idx]
                if artid not in vistos:
                    vistos[artid] = True
                    artids.append(artid)

            # 3 - Si el último resultado está dentro del umbral y quedan más chuncks, ampliar top_k
            if resultados and resultados[-1][0] <= self.semantic_threshold and top_k < len(self.chuncks):
                top_k *= 2
            else:
                # 4 - Salir: umbral superado o ya se han recuperado todos los chuncks
                break

        return artids
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # Se empieza con top_k=MAX_EMBEDDINGS y se dobla si el último resultado aún supera el
        # umbral. Esto es más eficiente en memoria que pedir len(chuncks) desde el principio:
        # en la mayoría de consultas bastará con los primeros MAX_EMBEDDINGS resultados, y solo
        # se amplía cuando hay evidencia de que quedan resultados relevantes sin recuperar.
        # Se usa un dict 'vistos' en lugar de set() para mantener el orden de inserción
        # (garantizado en Python 3.7+) y evitar duplicados: el mismo artículo puede aparecer
        # varias veces porque tiene varias frases indexadas, pero solo queremos contarlo una vez.
        # La condición de salida comprueba resultados[-1][0] (distancia del último resultado):
        # si ya supera el umbral, todos los siguientes también lo superarán (están más lejos).


    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """

        self.load_semantic_model()

        # 1 - Consultar el modelo con todos los chuncks para obtener el orden de relevancia completo
        resultados = self.model.query(query, top_k=len(self.chuncks))

        # Construir lookup de artids presentes en la lista de entrada (dict en lugar de set())
        en_articles = {a: True for a in articles}

        # 2, 3 y 4 - Recorrer resultados en orden de relevancia y añadir artids que estén en articles
        reordenados = []
        ya_añadidos = {}
        for dist, idx in resultados:
            artid = self.chunck_index[idx]
            if artid in en_articles and artid not in ya_añadidos:
                reordenados.append(artid)
                ya_añadidos[artid] = True

        return reordenados
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # A diferencia de solve_semantic_query, aquí se pide top_k=len(self.chuncks) desde el
        # principio porque necesitamos el orden de similitud COMPLETO para poder reordenar
        # todos los artículos de la lista 'articles'. Si usáramos MAX_EMBEDDINGS, podría
        # ocurrir que algunos artículos de 'articles' quedaran fuera del top_k y no aparecieran
        # en el resultado reordenado, perdiendo resultados de la búsqueda booleana original.
        # 'en_articles' es un dict de lookup O(1) en lugar de recorrer la lista 'articles'
        # en cada iteración (que sería O(n) por elemento, O(n²) en total).
        # 'ya_añadidos' evita que el mismo artículo aparezca varias veces (múltiples frases).
    

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
        # Si el modo semántico está activo, construir el kdtree tras indexar todos los ficheros
        if self.semantic:
            self.create_kdtree()
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article


    def index_file(self, filename:str):
        """

        Indexa el contenido de un fichero.

        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.positional se debe ampliar el indexado

        """

        # Asignamos un docid único e incremental a este fichero
        docid = len(self.docs)
        self.docs[docid] = filename

        for i, line in enumerate(open(filename, encoding='utf-8')):
            articulo = self.parse_article(line)

            # Ignoramos artículos cuya URL ya fue procesada (duplicados entre ficheros)
            if self.already_in_index(articulo):
                continue
            self.urls.add(articulo['url'])

            # Asignamos un artid único e incremental a este artículo.
            # Guardamos (docid, posición_dentro_del_fichero) para poder localizarlo.
            artid = len(self.articles)
            self.articles[artid] = (docid, i)

            # Tokenizamos el campo por defecto ('all') del artículo
            tokens = self.tokenize(articulo[self.DEFAULT_FIELD])

            if self.positional:
                # Modo posicional: guardamos TODAS las posiciones de cada token.
                # self.index[token] = {artid: [pos1, pos2, ...]} (dict)
                for pos, token in enumerate(tokens):
                    if token not in self.index:
                        self.index[token] = {}
                    if artid not in self.index[token]:
                        self.index[token][artid] = []
                    self.index[token][artid].append(pos)
            else:
                # Modo básico: una sola entrada por artid, sin posiciones.
                # self.index[token] = [artid1, artid2, ...] (lista ordenada)
                vistos = set()  # evita duplicar artid en la misma posting list
                for token in tokens:
                    if token not in vistos:
                        vistos.add(token)
                        if token not in self.index:
                            self.index[token] = []
                        self.index[token].append(artid)

            # Si el modo semántico está activo, añadir las frases del artículo al índice semántico
            if self.semantic:
                self.update_chuncks(articulo[self.DEFAULT_FIELD], artid)
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # El índice posicional usa dict {artid: [pos1, pos2, ...]} en lugar de lista plana
        # porque get_positionals necesita acceder en O(1) a las posiciones de un artículo
        # concreto. Con una lista plana habría que recorrerla entera para cada artículo.
        # El índice básico usa lista [artid1, artid2, ...] porque solo necesitamos saber
        # si el artículo contiene el término, sin importar dónde; la lista es más compacta.
        # Se usa un set 'vistos' local por artículo para evitar añadir el mismo artid dos veces
        # cuando un token aparece repetido; así la posting list queda sin duplicados.
        # self.articles[artid] = (docid, i) almacena la posición de línea para poder releer
        # el artículo original en solve_and_show sin cargar todo el fichero en memoria.
        # update_chuncks se llama aquí (no en index_dir) porque en este punto el texto del
        # artículo ya está parseado y disponible en 'articulo[self.DEFAULT_FIELD]'.


    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        print('='*40)
        print('Estadísticas del índice')
        print('='*40)
        print(f'Número de ficheros indexados : {len(self.docs)}')
        print(f'Número de artículos indexados: {len(self.articles)}')
        print(f'Número de términos en el índice: {len(self.index)}')
        print('='*40)



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query:str, prev:Dict={}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Parsea la consulta término a término de izquierda a derecha.
        - Varios términos implican AND implícito.
        - NOT delante de un término niega ese término antes de hacer el AND.

        param:  "query": cadena con la query
                "prev": incluido por si se quiere hacer una version recursiva.

        return: tupla (posting_list, query_original) con el resultado de la query

        """

        if query is None or len(query) == 0:
            return [], query

        # Usamos regex para tokenizar: las frases entre comillas se tratan
        # como un único token; NOT y los términos sueltos son el resto.
        TOKEN_RE = re.compile(r'"[^"]*"|NOT|\S+')
        tokens = TOKEN_RE.findall(query)

        resultado = None  # None significa "todavía no hay lista acumulada"
        negar_siguiente = False  # indica si el próximo término va precedido de NOT

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token.upper() == 'NOT':
                # El siguiente término va negado
                negar_siguiente = True
                i += 1
                continue

            # Obtenemos la posting list del término (en minúsculas)
            posting = self.get_posting(token.lower())

            # Aplicamos NOT si el término está negado
            if negar_siguiente:
                posting = self.reverse_posting(posting)
                negar_siguiente = False

            # AND implícito: intersectamos con el resultado acumulado
            if resultado is None:
                resultado = posting
            else:
                resultado = self.and_posting(resultado, posting)

            i += 1

        # Si la consulta solo era "NOT" sin término, devolvemos lista vacía
        if resultado is None:
            resultado = []

        return resultado, query
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # Se usa una expresión regular para tokenizar en lugar de split() porque necesitamos
        # tratar las frases entre comillas (p.ej. "machine learning") como un único token;
        # split() las partiría en tokens separados, perdiendo la información de frase.
        # El AND es implícito (varios términos sin operador = intersección) siguiendo el
        # comportamiento estándar de los motores de búsqueda tipo Google.
        # resultado=None (en vez de []) permite distinguir "aún no hay lista" de "lista vacía";
        # si empezáramos con [] y aplicáramos and_posting, el resultado siempre sería vacío.




    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Si el término está entre comillas delega en get_positionals.
        Soporta dos formatos de índice:
          - Lista [artid, ...]: índice básico (self.positional=False)
          - Dict {artid: [pos, ...]}: índice posicional (self.positional=True)

        param:  "term": termino del que se debe recuperar la posting list.
                        Puede ser una frase entre comillas: '"word1 word2"'

        return: posting list (lista ordenada de artids)

        NECESARIO PARA TODAS LAS VERSIONES

        """
        # Si el término es una frase entre comillas → búsqueda posicional
        if term.startswith('"') and term.endswith('"') and len(term) > 2:
            phrase_terms = self.tokenize(term[1:-1])  # quita comillas y tokeniza
            return self.get_positionals(phrase_terms)

        entrada = self.index.get(term, None)
        if entrada is None:
            return []
        # Índice posicional: el valor es un dict {artid: [posiciones]}
        if isinstance(entrada, dict):
            return sorted(entrada.keys())
        # Índice básico: el valor ya es una lista de artids
        return entrada
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # Se detecta el tipo de índice con isinstance(entrada, dict) en lugar de consultar
        # self.positional porque es más robusto: funciona aunque el índice se cargue desde
        # fichero y self.positional no esté correctamente restaurado.
        # Para el índice posicional se extraen y ordenan las claves del dict porque and_posting
        # y reverse_posting esperan listas ordenadas de artids.
        # La detección de frase (comillas) se hace aquí y no en solve_query porque get_posting
        # es el punto de entrada único para recuperar listas; así solve_query no necesita saber
        # si un token es término simple o frase.



    def get_positionals(self, terms:list):
        """

        Devuelve la posting list de los artículos donde los términos aparecen
        juntos y en orden CONSECUTIVO (búsqueda de frases).
        NECESARIO PARA LAS BÚSQUEDAS POSICIONALES

        Algoritmo:
          1. Recupera los dicts posicionales {artid: [pos, ...]} de cada término.
          2. Intersecta los artids con and_posting (merge, sin sets).
          3. Para cada artid común busca una posición de inicio en el primer
             término tal que term[k] aparezca en pos_inicio+k para todo k.

        param:  "terms": lista de términos consecutivos.

        return: posting list ordenada de artids donde la frase aparece.

        """
        if not terms:
            return []

        # Recuperamos los dicts posicionales de cada término
        postings = []
        for term in terms:
            entrada = self.index.get(term, {})
            # Si el índice no es posicional, no podemos resolver la frase
            if not isinstance(entrada, dict):
                return []
            postings.append(entrada)

        # Intersección de artids usando and_posting (merge, sin sets)
        artids = sorted(postings[0].keys())
        for k in range(1, len(postings)):
            artids = self.and_posting(artids, sorted(postings[k].keys()))

        # Verificamos consecutividad término a término.
        # Convertimos las listas de posiciones a sets para búsqueda O(1).
        resultado = []
        for artid in artids:
            posiciones_t0 = postings[0][artid]  # posiciones candidatas del 1er término
            # Precalculamos los sets de posiciones de los demás términos para este artid
            sets_pos = [set(postings[k][artid]) for k in range(1, len(terms))]

            frase_encontrada = False
            for pos_inicio in posiciones_t0:
                # Comprobamos que term[k] aparece en pos_inicio+k para todo k>=1
                if all((pos_inicio + k) in sets_pos[k - 1] for k in range(1, len(terms))):
                    frase_encontrada = True
                    break  # basta con encontrar una ocurrencia por artículo

            if frase_encontrada:
                resultado.append(artid)

        return resultado
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # Primero se intersectan los artids (and_posting) para reducir el conjunto candidato
        # antes de comprobar consecutividad. Así solo verificamos los artículos que contienen
        # TODOS los términos, evitando trabajo innecesario.
        # Las posiciones de los términos k>0 se convierten a set para que la comprobación
        # (pos_inicio+k) in sets_pos[k-1] sea O(1) en vez de O(posiciones) con lista.
        # Se usa and_posting (merge lineal O(n+m)) en lugar de set() para mantener la
        # consistencia con el resto del código y no usar sets en posting lists.



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Implementado con merge lineal (sin sets): recorre todos los artids existentes
        y se queda con los que NO aparecen en p.

        param:  "p": posting list a negar

        return: posting list con todos los artid excepto los de p

        """
        resultado = []
        # Los artids van de 0 a len(self.articles)-1 y están ordenados
        total = len(self.articles)
        j = 0  # puntero sobre p

        for artid in range(total):
            # Avanzamos j hasta que p[j] >= artid
            while j < len(p) and p[j] < artid:
                j += 1
            # Si p[j] != artid (o ya terminó p), este artid NO está en p
            if j >= len(p) or p[j] != artid:
                resultado.append(artid)

        return resultado
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # Se implementa como merge lineal O(n) en lugar de convertir p a set() y hacer
        # 'if artid not in p_set' porque la coherencia con el resto del sistema lo exige:
        # las posting lists son listas ordenadas y todas las operaciones booleanas deben
        # mantener ese invariante sin usar sets (requisito del enunciado del proyecto).
        # Se recorre range(total) en vez de iterar sobre self.articles.keys() para garantizar
        # que el universo es exactamente [0, total-1] sin huecos, lo cual es cierto porque
        # los artids se asignan de forma incremental en index_file.



    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting lists de forma EFICIENTE usando el
        algoritmo de merge con dos punteros. Complejidad O(|p1| + |p2|).

        param:  "p1", "p2": posting lists sobre las que calcular

        return: posting list con los artid incluidos en p1 y p2

        """
        resultado = []
        i = 0  # puntero sobre p1
        j = 0  # puntero sobre p2

        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                # Elemento común: lo añadimos al resultado
                resultado.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                # p1 va por detrás, lo avanzamos
                i += 1
            else:
                # p2 va por detrás, lo avanzamos
                j += 1

        return resultado
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # El algoritmo de merge con dos punteros es O(|p1|+|p2|), óptimo para listas ordenadas.
        # Alternativas descartadas:
        #   - Conversión a set() e intersección: O(n) construcción + O(min) intersección,
        #     pero rompe el invariante de lista ordenada y no permite reutilizar and_posting
        #     en cadena (solve_query lo llama repetidamente acumulando resultados).
        #   - Búsqueda binaria (bisect): O(n·log m), peor que merge cuando las listas son
        #     de tamaño similar.
        # Las posting lists siempre están ordenadas por artid creciente (se insertan en orden
        # en index_file), lo que garantiza la corrección del merge.






    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el EXCEPT de dos posting lists de forma EFICIENTE.
        Algoritmo de merge con dos punteros: O(|p1| + |p2|).

        param:  "p1", "p2": posting lists sobre las que calcular

        return: posting list con los artid de p1 que NO están en p2

        """
        resultado = []
        i = 0  # puntero sobre p1
        j = 0  # puntero sobre p2

        while i < len(p1):
            # Avanzamos j hasta que p2[j] >= p1[i]
            while j < len(p2) and p2[j] < p1[i]:
                j += 1
            # Si p2 se acabó o p2[j] != p1[i], el elemento de p1 no está en p2
            if j >= len(p2) or p2[j] != p1[i]:
                resultado.append(p1[i])
            i += 1

        return resultado
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # Mismo principio que and_posting: merge lineal O(|p1|+|p2|) sobre listas ordenadas.
        # La diferencia con and_posting es que cuando p1[i] < p2[j] (o p2 agotada), el
        # elemento sí se incluye (está en p1 pero no en p2). Si p1[i] == p2[j], se descarta.
        # Se avanza j solo cuando p2[j] < p1[i], no cuando son iguales, para no perder
        # posibles igualdades con el siguiente elemento de p1.





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados.
        Muestra como máximo self.SHOW_MAX resultados salvo que self.show_all sea True.

        param:  "query": query que se debe resolver.

        return: el numero de artículos recuperados

        """
        resultado, _ = self.solve_query(query)
        n = len(resultado)

        print(f'Query: "{query}" --> {n} resultado(s)')

        # Determinamos cuántos resultados mostrar
        mostrar = resultado if self.show_all else resultado[:self.SHOW_MAX]

        for artid in mostrar:
            # Recuperamos el docid y la posición del artículo dentro del fichero
            docid, pos = self.articles[artid]
            filename = self.docs[docid]

            # Leemos la línea correspondiente del fichero para obtener título y URL
            with open(filename, encoding='utf-8') as fh:
                for idx, line in enumerate(fh):
                    if idx == pos:
                        art = self.parse_article(line)
                        break

            print(f'  #{artid}\t{art["title"]}\t{art["url"]}')

        return n
        # -------------------------------------------------------
        # DECISIONES DE DISEÑO:
        # No se almacena el contenido completo del artículo en self.articles para no duplicar
        # en memoria los datos ya presentes en los ficheros JSON. En su lugar se guarda (docid, pos)
        # y se relee el fichero bajo demanda en show: esto es eficiente porque show solo se
        # llama con SHOW_MAX resultados (10 por defecto), no con miles.
        # Se usa enumerate(fh) en vez de fh.readlines()[pos] para no cargar todo el fichero
        # en memoria; se para en cuanto se encuentra la línea buscada (break).



