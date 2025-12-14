# services/ingest/main.py
from __future__ import annotations

import pathlib
from typing import List

import nltk
from nltk.tokenize import sent_tokenize

from config import get_settings
from services.rag.embeddings import embed_text
from services.rag.vector_store import VectorStore, VectorStoreConfig


# Tokenizer
nltk.download("punkt")

def load_cv_text(path: str) -> str:
    cv_path = pathlib.Path(path)
    if not cv_path.exists():
        raise FileNotFoundError(f"No se encontró el CV en {cv_path.resolve()}")

    return cv_path.read_text(encoding="utf-8")


def chunk_text(
    text: str,
    max_chars: int = 400,
) -> List[str]:
    """
    Chunking simple basado en oraciones (NLTK) para no cortar frases a la mitad.
    """
    sentences = sent_tokenize(text, language="spanish")
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    settings = get_settings()

    print("Cargando CV...")
    raw_text = load_cv_text('data/cv_jose_perez.txt')
    print("Generando chunks...")
    chunks = chunk_text(raw_text, max_chars=2000)
    print(f"Total de chunks generados: {len(chunks)}")

    print("Creando embeddings...")
    embeddings = embed_text(
        chunks,
        model_name=settings.embedding_model_name,
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError("Cantidad de embeddings y chunks no coincide")

    ids = [f"chunk-{i}" for i in range(len(chunks))]
    metadatas = [{"text": chunk} for chunk in chunks]

    print("Conectando a Pinecone...")
    vs_config = VectorStoreConfig(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        dimension=len(embeddings[0]),
    )
    store = VectorStore(vs_config)

    print("Upsert de vectores en Pinecone...")
    store.upsert(ids=ids, vectors=embeddings, metadatas=metadatas)

    print("Ingesta completa. El índice está listo para el chatbot.")


if __name__ == "__main__":
    main()
