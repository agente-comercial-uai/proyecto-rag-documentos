"""Almacen de vectores con FAISS (archivo local).

Implementacion por defecto porque Redis nativo de Windows no incluye
RediSearch (no soporta vectores). FAISS guarda el indice en disco y no
requiere servidores externos.

Si en el futuro se tiene Redis con RediSearch, solo hay que:
1) instalar `redisvl` y `langchain-redis`,
2) cambiar VECTOR_BACKEND=redis y REDIS_URL en el entorno,
sin tocar el resto del codigo (el agente y la herramienta solo usan
la interfaz `buscar(texto, k)`).
"""

from __future__ import annotations

import os
import pickle

import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "faiss_index")
_INDICE = os.path.join(_DIR, "indice.faiss")
_DATOS = os.path.join(_DIR, "datos.pkl")


class VectorStore:
    """Interfaz unica de busqueda. Backend FAISS por defecto."""

    def __init__(self, indice: faiss.Index, datos: list[dict], embeddings):
        self.indice = indice
        self.datos = datos
        self.embeddings = embeddings

    def buscar(self, texto: str, k: int = 3) -> list[dict]:
        """Devuelve los k trozos mas parecidos con su archivo, texto y score.

        El score es la similitud coseno (IndexFlatIP sobre vectores
        normalizados), con rango [-1, 1]: 1 = identico, 0 = ortogonal,
        -1 = opuesto.
        """
        v = np.array(self.embeddings.embed_query(texto), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(v)
        distancias, ids = self.indice.search(v, k)
        resultados = []
        for pos, idx in enumerate(ids[0]):
            if 0 <= idx < len(self.datos):
                resultados.append({**self.datos[idx], "score": float(distancias[0][pos])})
        return resultados


def _embeddings():
    modelo = os.getenv("EMBEDDING_MODELO", "gemini-embedding-001")
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY no esta configurada en .env")
    return GoogleGenerativeAIEmbeddings(model=f"models/{modelo}", google_api_key=api_key)


def _chunkear(texto: str, tamano: int = 400, solape: int = 50) -> list[str]:
    """Parte un texto en trozos de ~tamano caracteres con solape."""
    trozos = []
    inicio = 0
    while inicio < len(texto):
        trozo = texto[inicio : inicio + tamano].strip()
        if trozo:
            trozos.append(trozo)
        inicio += tamano - solape
    return trozos


def _cargar_documentos(carpeta: str) -> list[dict]:
    """Lee los .txt de la carpeta y devuelve (nombre, trozo)."""
    registros = []
    for nombre in sorted(os.listdir(carpeta)):
        if not nombre.endswith(".txt"):
            continue
        ruta = os.path.join(carpeta, nombre)
        with open(ruta, "r", encoding="utf-8") as f:
            contenido = f.read()
        for trozo in _chunkear(contenido):
            registros.append({"archivo": nombre, "texto": trozo})
    return registros


def construir_indice(carpeta: str) -> VectorStore:
    """(Re)construye el indice FAISS y lo guarda en disco."""
    emb = _embeddings()
    registros = _cargar_documentos(carpeta)

    if not registros:
        raise RuntimeError("No se encontraron documentos .txt en " + carpeta)

    textos = [r["texto"] for r in registros]
    vectores = np.array(emb.embed_documents(textos), dtype="float32")
    faiss.normalize_L2(vectores)

    dimension = vectores.shape[1]
    indice = faiss.IndexFlatIP(dimension)
    indice.add(vectores)

    os.makedirs(_DIR, exist_ok=True)
    faiss.write_index(indice, _INDICE)
    with open(_DATOS, "wb") as f:
        pickle.dump(registros, f)

    return VectorStore(indice, registros, emb)


def cargar_indice() -> VectorStore:
    """Carga el indice FAISS guardado. Si no existe, lo construye."""
    if not (os.path.exists(_INDICE) and os.path.exists(_DATOS)):
        base = os.path.abspath(os.path.join(_DIR, "..", "documentos"))
        return construir_indice(base)

    indice = faiss.read_index(_INDICE)
    with open(_DATOS, "rb") as f:
        datos = pickle.load(f)
    return VectorStore(indice, datos, _embeddings())
