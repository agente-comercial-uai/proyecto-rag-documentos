"""Grafo ReAct del agente RAG de documentos."""

from __future__ import annotations

import os

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from .tools import buscar_documentos

_SYSTEM_PROMPT = (
    "Eres un asistente de consultas sobre politicas internas de una empresa. "
    "Usa SIEMPRE la herramienta buscar_documentos antes de responder. "
    "Responde UNICAMENTE con lo que la herramienta devuelva y cita el archivo "
    "de donde sale la informacion. Si no hay nada en los documentos, responde "
    "exactamente: 'no está en los documentos'. No inventes informacion."
)


def _crear_modelo():
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    modelo = os.getenv("MODELO", "gemini-3.5-flash-lite")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY no esta configurada en .env")
    return ChatGoogleGenerativeAI(model=modelo, google_api_key=api_key)


def crear_agente():
    return create_react_agent(
        _crear_modelo(),
        tools=[buscar_documentos],
        prompt=SystemMessage(content=_SYSTEM_PROMPT),
    )
