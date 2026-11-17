"""Punto de entrada del agente RAG de documentos.

Uso:
    python main.py documentos "pregunta" [--saltos] [--diagrama]

Ejemplos:
    python main.py documentos "Cuantos dias de vacaciones tengo?" --saltos
    python main.py documentos "Cual es el bono por trabajar con mas de 40C de temperatura?"
    python main.py documentos --diagrama
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def _aplanar_contenido(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        partes = []
        for bloque in contenido:
            if isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(bloque.get("text", ""))
        return "".join(partes)
    return ""


def _texto_final(mensajes) -> str:
    for m in reversed(mensajes):
        texto = _aplanar_contenido(getattr(m, "content", "") or "")
        if texto.strip():
            return texto
    return ""


def _mostrar_saltos(mensajes) -> int:
    saltos = 0
    for m in mensajes:
        tipo = getattr(m, "type", "")
        if tipo == "human":
            continue
        saltos += 1
        tool_calls = getattr(m, "tool_calls", None)
        etiqueta = f"[{tipo}]"
        if tool_calls:
            for tc in tool_calls:
                nombre = tc.get("name", "?")
                args = tc.get("args", {})
                print(f"{etiqueta} decidio llamar a `{nombre}` con {args}")
        elif tipo == "tool":
            print(f"{etiqueta} recupero:\n{_aplanar_contenido(getattr(m, 'content', ''))}")
        else:
            print(f"{etiqueta} {_aplanar_contenido(getattr(m, 'content', ''))}")
    return saltos


def _ejecutar(mensaje: str, ver_saltos: bool) -> None:
    from langchain_core.messages import HumanMessage

    from src.documentos.graph import crear_agente

    agente = crear_agente()
    estado_final = agente.invoke({"messages": [HumanMessage(content=mensaje)]})
    mensajes = estado_final.get("messages", [])

    if ver_saltos:
        total = _mostrar_saltos(mensajes)
        print(f"\n-- total de pasos del agente: {total} --\n")

    print(_texto_final(mensajes))


def _diagrama() -> None:
    from src.documentos.graph import crear_agente

    agente = crear_agente()
    print(agente.get_graph().draw_mermaid())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Agente RAG de documentos.")
    parser.add_argument("agente", help="Nombre del agente (documentos)")
    parser.add_argument("mensaje", nargs="?", help="Mensaje para el agente")
    parser.add_argument("--saltos", action="store_true", help="Muestra cada paso del razonamiento.")
    parser.add_argument("--diagrama", action="store_true", help="Imprime el grafo Mermaid.")
    args = parser.parse_args(argv)

    if args.diagrama:
        _diagrama()
        return 0

    if not args.mensaje:
        print("Debes indicar un mensaje, por ejemplo: python main.py documentos \"hola\"")
        return 2

    _ejecutar(args.mensaje, args.saltos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
