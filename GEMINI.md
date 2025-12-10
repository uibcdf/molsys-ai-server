# Guía para el Asistente Gemini

Este documento proporciona un contexto rápido y una guía de operaciones para trabajar eficientemente en el repositorio `molsys-ai`.

## 1. Resumen del Proyecto

**Objetivo:** Construir un asistente de IA para el ecosistema de herramientas de simulación molecular `MolSys*` (MolSysMT, MolSysViewer, etc.).

**Componentes Clave:**
1.  **Agente Autónomo CLI:** Un agente que utiliza las herramientas de MolSysSuite para realizar flujos de trabajo.
2.  **Chatbot de Documentación:** Un chatbot basado en RAG para responder preguntas sobre el uso de las herramientas, integrable en la documentación web.
3.  **Modelo de Lenguaje Re-entrenado:** Un LLM especializado en el conocimiento del ecosistema MolSys*.

## 2. Entorno de Desarrollo

**¡IMPORTANTE!** El entorno de trabajo se gestiona exclusivamente con **Conda** a través del archivo `environment.yml`.

- **Dependencias Críticas:** Herramientas como `molsysmt` **no están en PyPI**. Se distribuyen a través del canal `uibcdf` de Conda, que ya está configurado en `environment.yml`.
- **Instalación/Actualización:** Para configurar el entorno, utiliza:
  ```bash
  # Si el entorno no existe
  conda env create -f environment.yml

  # Si el entorno ya existe y quieres actualizarlo
  conda env update --file environment.yml --name molsys-ai
  ```
- **Modo Editable:** Para que los comandos como `molsys-ai` estén disponibles, el paquete debe instalarse en modo editable. Después de activar el entorno, ejecuta:
  ```bash
  pip install -e .
  ```

## 3. Comandos de Ejecución Clave

- **Servidor del Modelo (Stub):**
  ```bash
  uvicorn model_server.server:app --reload
  ```
- **CLI (apuntando al servidor):**
  ```bash
  molsys-ai --server-url http://127.0.0.1:8000 --message "Tu pregunta"
  ```
- **Backend del Chat de Documentación:**
  ```bash
  uvicorn docs_chat.backend:app --reload
  ```

## 4. Pruebas (Testing)

- **Ejecutar Pruebas:**
  ```bash
  pytest
  ```
- **Convención para Mocks:** Para evitar el error `ModuleNotFoundError: No module named 'molsysmt'`, cualquier prueba que directa o indirectamente importe `molsysmt` debe simular (mock) el módulo. La convención establecida en este proyecto es parchear `sys.modules`. **Observa `tests/test_smoke.py` como ejemplo de referencia**:
  ```python
  # Al principio de la función de prueba
  mocker.patch.dict("sys.modules", {"molsysmt": mocker.Mock()})

  # Importa los módulos que dependen de molsysmt DESPUÉS del parche
  from agent.core import MolSysAIAgent
  ```

## 5. Sistema RAG (Retrieval-Augmented Generation)

- **Propósito:** Es el motor de búsqueda de conocimiento para el chatbot. Se basa en encontrar fragmentos de documentación relevantes para una pregunta.
- **Fuente de Datos:** La documentación del repositorio `molsysmt`, que se asume que está en `../molsysmt/`.
- **Índice:** El proceso de RAG genera un índice (`data/rag_index.pkl`) que contiene los textos y sus embeddings vectoriales.
- **Construcción del Índice:** Para generar o actualizar el índice, se debe ejecutar un script que llame a `rag.build_index.build_index()`. El script temporal `_build_index_script.py` se creó para este fin:
  ```bash
  python _build_index_script.py
  ```
  **Nota:** La primera vez, este comando descarga un modelo de `sentence-transformers` y luego procesa la documentación. Puede tardar varios minutos y utilizará la GPU si está disponible.

## 6. Estado Actual y Siguientes Pasos

Para conocer el estado exacto en el que se dejó el proyecto y cuáles son los siguientes pasos inmediatos, **consulta siempre el archivo `checkpoint.md`**. Este archivo se actualiza al final de cada sesión de trabajo para garantizar una transición fluida.
