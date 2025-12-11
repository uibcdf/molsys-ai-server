# Checkpoint: Estado del Proyecto MolSys-AI

Este documento resume el estado actual del proyecto, los objetivos, el progreso realizado y los siguientes pasos para facilitar la transición a una nueva máquina de trabajo.

## 1. Objetivos Globales (Qué queremos)

Los tres objetivos principales a largo plazo para este proyecto son:

1.  **Re-entrenamiento del Modelo:** Re-entrenar un modelo de lenguaje para que tenga un conocimiento profundo de las herramientas del ecosistema MolSysSuite.
2.  **Chatbot de Documentación:** Integrar un chatbot en las páginas web de la documentación (generadas con Sphinx) de `molsysmt`, `molsysviewer`, etc., para responder preguntas sobre el uso de las herramientas.
3.  **Agente Autónomo CLI:** Desarrollar un agente de línea de comandos (`molsys-ai`) capaz de ejecutar flujos de trabajo autónomos utilizando las herramientas de MolSysSuite a petición del usuario.

## 2. Progreso Realizado (Qué hemos hecho)

Hemos completado la fase inicial de configuración y la implementación de la primera funcionalidad clave del agente.

### Fase 1: Esqueleto Funcional y Verificación de Conexiones

- **Confirmamos la conexión `CLI -> Agente -> Servidor`:** Verificamos que los tres componentes principales del sistema pueden comunicarse entre sí.
- **`model_server`:** Se puede iniciar correctamente.
- **`cli`:** El comando `molsys-ai` puede enviar un mensaje al servidor a través del agente.

### Fase 2: Implementación de la Primera Herramienta (Tool-Calling)

- **Propósito:** Construir la "columna vertebral" del agente autónomo, implementando el ciclo completo de planificación y ejecución de una herramienta.
- **`agent/tools/molsysmt_tools.py`:** Se implementó una herramienta real (`get_info`) que obtiene información de un PDB ID.
- **`agent/planner.py`:** Se mejoró el planificador para detectar la intención del usuario de usar la nueva herramienta y para generar un plan de ejecución.
- **`agent/core.py`:** Se actualizó el agente para ejecutar el plan del planificador, llamar al `executor` de herramientas y utilizar el resultado para formular una respuesta.
- **`tests/test_smoke.py`:** Se añadió una prueba que verifica el flujo de "tool-calling" de extremo a extremo, utilizando "mocks" para evitar la dependencia real de `molsysmt` durante las pruebas.

### Fase 3: Preparación del Sistema RAG (Retrieval-Augmented Generation)

- **Propósito:** Sentar las bases para el chatbot de documentación (objetivo #2) y la futura generación de datos de entrenamiento (objetivo #1).
- **Dependencias:** Se añadieron `sentence-transformers` y `numpy` al archivo `environment.yml` para gestionar las dependencias del entorno `conda`.
- **`rag/embeddings.py`:** Se implementó un modelo de embeddings real (`all-MiniLM-L6-v2`) para convertir texto en vectores.
- **`rag/build_index.py`:** Se implementó la lógica para encontrar archivos de documentación (`.md`), dividirlos en trozos, generar sus embeddings y guardarlos en un archivo de índice (`.pkl`).
- **`rag/retriever.py`:** Se implementó la lógica de búsqueda vectorial (similitud de coseno) para encontrar los documentos más relevantes para una consulta dada.
- **Integración:** Se actualizaron `agent/planner.py` y `docs_chat/backend.py` para ser compatibles con la nueva lógica de carga y consulta del índice RAG.

### Fase 4: Construcción del Índice RAG y Ajuste del Entorno

- **Propósito:** Generar el índice vectorial para el sistema RAG y asegurar que el entorno de ejecución sea estable.
- **`_build_index_script.py`:** Se creó y ejecutó con éxito el script para procesar la documentación de `../molsysmt`.
- **Índice RAG:** Se generó y guardó el índice en `data/rag_index.pkl`, conteniendo 1227 fragmentos de texto con sus correspondientes embeddings.
- **Ajuste del Entorno:** Se identificó y resolvió un conflicto de dependencias entre `pyarrow` y `datasets`. Se fijó la versión `datasets=4.4.1` en `environment.yml` para garantizar la reproducibilidad del entorno.

## 3. Plan General (A dónde vamos)

La estrategia acordada es centrarse en la **implementación completa del sistema RAG** como el siguiente gran paso, ya que es fundamental para los tres objetivos globales.

El flujo de trabajo RAG que estamos construyendo es:

1.  **Indexación (fuera de línea):** Leer toda la documentación de MolSysMT, dividirla en fragmentos, calcular sus embeddings y guardar todo en un índice persistente. **(Completado)**
2.  **Consulta (en tiempo real):** Cuando un usuario hace una pregunta, el sistema calcula el embedding de la pregunta, busca en el índice los fragmentos de texto más similares y los recupera.
3.  **Generación de Respuesta:** El agente o chatbot recibe la pregunta original junto con el contexto recuperado ("documentación relevante") y utiliza esta información para generar una respuesta precisa y fundamentada.

Este sistema será el núcleo del chatbot de documentación y una fuente de datos invaluable para el re-entrenamiento del modelo.

## 4. Siguientes Pasos Inmediatos

Hemos completado con éxito la construcción del índice RAG y ajustado el entorno. El siguiente paso es verificar que el sistema funciona de extremo a extremo.

1.  **Probar el Sistema RAG (Ejecución de Servidores y Consulta):**
    Para una prueba completa del sistema RAG, necesitamos dos servidores Uvicorn corriendo simultáneamente en dos terminales separadas:

    *   **Terminal 1 (Servidor del Modelo - Puerto 8000):**
        Inicia el servidor del modelo. Este componente actúa como el "cerebro" del chatbot, al que el backend de `docs_chat` enviará las preguntas.
        ```bash
        uvicorn model_server.server:app --reload
        ```
        (Asegúrate de que esta terminal muestre "Application startup complete" y no la cierres).

    *   **Terminal 2 (Backend de Chat de Documentación - Puerto 8001):**
        Inicia el backend del chat de documentación. Este componente utiliza el índice RAG que acabamos de construir y se comunica con el servidor del modelo. Es crucial especificar la variable de entorno `MOLSYS_AI_DOCS_INDEX` para que cargue el índice correcto, y usar un puerto diferente para evitar conflictos.
        ```bash
        MOLSYS_AI_DOCS_INDEX=data/rag_index.pkl uvicorn docs_chat.backend:app --reload --port 8001
        ```
        (Asegúrate de que esta terminal muestre "Index loaded with 1227 documents" y "Application startup complete").

    Una vez que ambos servidores estén corriendo, podemos realizar una consulta de prueba.

2.  **Realizar una Consulta de Prueba (`curl`):**
    Abre una tercera terminal y ejecuta el siguiente comando `curl` para enviar una pregunta al backend de chat de documentación. Este comando consultará el índice RAG y enviará la consulta al servidor del modelo.
    ```bash
    curl -X POST http://127.0.0.1:8001/v1/docs-chat \
         -H "Content-Type: application/json" \
         -d '{"query": "How to load a PDB ID in MolSysMT?", "k": 3}'
    ```
    La respuesta de este comando `curl` nos indicará si el sistema RAG y la integración con el modelo funcionan correctamente.

3.  **Limpiar el repositorio:** Eliminar el script temporal `_build_index_script.py`. (Ya realizado)