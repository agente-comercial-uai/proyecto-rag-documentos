"""Herramientas del agente RAG de documentos."""

from __future__ import annotations

from langchain_core.tools import tool

from . import vectorstore

_STORE = None


def _store():
    global _STORE
    if _STORE is None:
        _STORE = vectorstore.cargar_indice()
    return _STORE


@tool
def buscar_documentos(pregunta: str) -> str:
    """Busca en los documentos internos los trozos mas parecidos a `pregunta`.

    Devuelve los trozos con el nombre del archivo de origen. Si no hay nada
    relevante, responde exactamente "no está en los documentos".
    """
    resultados = _store().buscar(pregunta, k=3)

    # Umbral simple: el indice usa producto interno normalizado (1 = identico).
    relevantes = [r for r in resultados if r.get("score", 1.0) >= 0.0]

    if not relevantes:
        return "no está en los documentos"

    lineas = []
    for r in relevantes:
        lineas.append(f"[{r['archivo']}] {r['texto']}")
    return "\n\n".join(lineas)
