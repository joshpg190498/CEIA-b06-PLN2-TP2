# RAG CV Chatbot – TP2
**Chatbot con Retrieval-Augmented Generation (RAG) usando Streamlit, Pinecone, Groq, NLTK y Embeddings**

Este proyecto implementa un chatbot que responde preguntas exclusivamente utilizando la información contenida en un **CV personal** mediante la técnica de **Retrieval-Augmented Generation (RAG)**.  

El sistema ingesta, vectoriza, almacena y recupera fragmentos del CV desde Pinecone, y genera respuestas controladas mediante modelos LLaMA 3.1 alojados en Groq. La interfaz del chatbot está construida con Streamlit.


### Funcionalidades principales
- Indexación del CV mediante embeddings.
- Chunking del texto con soporte para español (NLTK).
- Almacenamiento vectorial en Pinecone.
- Recuperación de fragmentos relevantes vía búsqueda semántica.
- Generación de respuestas con Groq (LLaMA 3.1).
- Respuestas estrictamente basadas en el contenido real del CV.


### Arquitectura del Proyecto

```
.
├── app.py                        # Entrada principal de Streamlit
├── config.py                     # Configuración global
├── data/
│   └── cv_jose_perez.txt         # Base de conocimiento
├── services/
│   ├── rag/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── chatbot.py
│   ├── ingest/
│   │   └── main.py
│   └── streamlit/
│       └── main.py
└── .env
```

### Tecnologías utilizadas

| Componente | Tecnología |
|-----------|------------|
| **Vector DB** | Pinecone |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`) |
| **LLM** | Groq – LLaMA 3.1 |
| **NLP** | NLTK (tokenización en español) |
| **Frontend** | Streamlit |
| **Entorno** | uv |



### Instalación y configuración

### 1. Instalar dependencias
```bash
uv sync
```

### 2. Crear archivo `.env`
```env
PINECONE_API_KEY=tu_api_key
PINECONE_INDEX_NAME=cv-jose-perez-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
GROQ_API_KEY=tu_api_key
TOP_K=6
```

---

### Ingestar el CV (construir el índice)

```bash
uv run python -m services.ingest.main
```

Este proceso:
- Segmenta el CV en chunks,
- Genera embeddings,
- Los envía a Pinecone.

Repetir si el CV cambia.

---

### Ejecutar el chatbot

```bash
uv run streamlit run app.py
```

Disponible en:

```
http://localhost:8501
```

Incluye:
- Preguntas estilo chat,
- Respuestas generadas por Groq,
- Panel que muestra los fragmentos del CV usados como contexto.

---

### Cómo funciona el RAG en este proyecto

1. **Chunking del CV**  
   División por párrafos y tokenización en español (NLTK).

2. **Vectorización**  
   Cada fragmento se transforma en un embedding.

3. **Indexación en Pinecone**  
   Los vectores y metadatos se almacenan y quedan listos para ser consultados.

4. **Búsqueda semántica**  
   La pregunta del usuario se vectoriza y se buscan los `top_k` chunks más relevantes.

5. **Generación controlada**  
   El LLM de Groq genera la respuesta **solo** usando el contexto recuperado.

### Comentarios
- Aumentar `top_k` o ajustarlo dinámicamente según la pregunta.
- Agregar un chunk adicional con el CV completo como fallback.
- Implementar re-ranqueo (cross-encoder).
- Mejorar la separación de secciones del CV para obtener chunks más semánticos.

### Autor
**José Luis Perez Galindo**  