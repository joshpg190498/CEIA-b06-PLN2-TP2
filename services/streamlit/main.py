# services/streamlit/main.py
from __future__ import annotations

import streamlit as st

from config import get_settings
from services.rag.chatbot import RAGChatbot
from services.rag.vector_store import VectorStore, VectorStoreConfig


@st.cache_resource
def get_chatbot() -> RAGChatbot:
    settings = get_settings()

    vs_config = VectorStoreConfig(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        # dim se determina automáticamente al crear índice en ingest,
        # aquí no lo necesitamos.
    )
    store = VectorStore(vs_config)

    return RAGChatbot(
        vector_store=store,
        groq_api_key=settings.groq_api_key,
        embedding_model_name=settings.embedding_model_name,
        top_k=settings.top_k,
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # lista de dicts {role, content, chunks?}


def render_sidebar() -> None:
    st.sidebar.title("ℹ️ Info del TP")
    st.sidebar.markdown(
        """
**TP1 – Chatbot RAG sobre CV**

- Vector DB: Pinecone  
- LLM: Groq (LLaMA 3)  
- Embeddings: SentenceTransformers  
- NLP: NLTK (tokenización / chunking)  
- Frontend: Streamlit  

Ejemplos de preguntas:
- ¿Qué experiencia tiene José en ML?
- ¿Qué lenguajes de programación domina José?
- ¿Qué roles tuvo José en Ferreyros?
- ¿Qué estudió josé?
"""
    )


def main() -> None:
    st.set_page_config(
        page_title="Chatbot RAG",
        page_icon="🤖",
    )

    render_sidebar()
    init_session_state()
    chatbot = get_chatbot()

    st.title("🤖 Chatbot RAG")
    st.write(
        "Este asistente utiliza **RAG (Retrieval-Augmented Generation)** para responder "
        "preguntas usando el contenido del CV como base de conocimiento."
    )

    # Mostrar historial de conversación
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Si tiene contexto, lo mostramos en un expander
            if msg["role"] == "assistant" and msg.get("chunks"):
                with st.expander("Ver fragmentos del CV usados como contexto"):
                    for i, c in enumerate(msg["chunks"], start=1):
                        st.markdown(f"**Fragmento {i}** (score: {c.score:.3f})")
                        st.write(c.text)

    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta sobre el CV...")
    if user_input:
        # Mostrar mensaje del usuario
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generar respuesta usando RAG
        with st.chat_message("assistant"):
            with st.spinner("Espere respuesta..."):
                answer, chunks = chatbot.answer(user_input)
                st.markdown(answer)

                with st.expander("Ver fragmentos del CV usados como contexto"):
                    for i, c in enumerate(chunks, start=1):
                        st.markdown(f"**Fragmento {i}** (score: {c.score:.3f})")
                        st.write(c.text)

        # Guardar en historial
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "chunks": chunks,
            }
        )


if __name__ == "__main__":
    main()
