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

    # Umbral de similitud coseno: solo trozos con score >= 0.70 se consideran
    # relevantes. Calibrado con los 3 documentos: las preguntas que tienen
    # respuesta puntuan 0.77-0.81, las que no, ~0.64.
    relevantes = [r for r in resultados if r.get("score", 0.0) >= 0.70]

    if not relevantes:
        return "no está en los documentos"

    lineas = []
    for r in relevantes:
        lineas.append(f"[{r['archivo']}] {r['texto']}")
    return "\n\n".join(lineas)
