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
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
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

        #1 - completar

        #2 - completar

        pass             
        

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")
	    # completar
        print("done!")


        
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
        
        # COMPLETAR

        # 1
        # 2
        # 3
        # 4
        # 5


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
        # COMPLETAR
        # 1
        # 2
        # 3
        # 4
    

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



