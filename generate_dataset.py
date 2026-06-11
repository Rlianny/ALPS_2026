"""
Generate the synthetic learning resource dataset.

Domain: Artificial Intelligence and Machine Learning (21 resources, 4 depth levels).

Why this dataset satisfies the problem conditions:
  - Real, non-trivial dependencies (multi-level DAG, not a simple chain)
  - Heterogeneous durations (2h to 7h), making the time constraint genuinely restrictive
  - Rich Spanish-language descriptions for meaningful LLM utility scoring
  - 13 resources have multiple prerequisites, creating real selection tension
  - Total: 90h across all resources; typical budget (15-30h) forces non-trivial trade-offs
"""

import json


resources = [
    # ── Level 0: no prerequisites ──────────────────────────────────────────
    {
        "id": "python_basics",
        "name": "Python básico",
        "description": (
            "Sintaxis fundamental de Python: variables, listas, diccionarios, "
            "funciones, clases y manejo de archivos. Base para cualquier trabajo "
            "en ciencia de datos e IA."
        ),
        "duration_hours": 3,
        "prerequisites": []
    },
    {
        "id": "linear_algebra",
        "name": "Álgebra lineal",
        "description": (
            "Vectores, matrices, multiplicación matricial, transformaciones lineales, "
            "valores y vectores propios. Fundamento matemático de redes neuronales "
            "y reducción de dimensionalidad."
        ),
        "duration_hours": 5,
        "prerequisites": []
    },
    {
        "id": "calculus",
        "name": "Cálculo diferencial",
        "description": (
            "Derivadas, regla de la cadena, gradientes y optimización por descenso "
            "del gradiente. Esencial para entender cómo aprenden los modelos."
        ),
        "duration_hours": 4,
        "prerequisites": []
    },
    {
        "id": "probability",
        "name": "Probabilidad y estadística",
        "description": (
            "Distribuciones de probabilidad, esperanza, varianza, teorema de Bayes "
            "y conceptos de inferencia estadística. Base para modelos probabilísticos."
        ),
        "duration_hours": 4,
        "prerequisites": []
    },
    {
        "id": "discrete_math",
        "name": "Lógica y matemática discreta",
        "description": (
            "Proposiciones, lógica de predicados, conjuntos, grafos y combinatoria. "
            "Útil para sistemas expertos, razonamiento automático y algoritmos de IA."
        ),
        "duration_hours": 3,
        "prerequisites": []
    },

    # ── Level 1: require level-0 foundations ──────────────────────────────
    {
        "id": "numpy_pandas",
        "name": "NumPy y Pandas",
        "description": (
            "Manipulación eficiente de arrays numéricos con NumPy y análisis de datos "
            "tabulares con Pandas. Herramientas cotidianas en cualquier pipeline de ML."
        ),
        "duration_hours": 3,
        "prerequisites": ["python_basics"]
    },
    {
        "id": "data_visualization",
        "name": "Visualización de datos",
        "description": (
            "Creación de gráficas informativas con Matplotlib y Seaborn para explorar "
            "distribuciones, correlaciones y resultados de modelos."
        ),
        "duration_hours": 2,
        "prerequisites": ["python_basics", "numpy_pandas"]
    },
    {
        "id": "classical_ml",
        "name": "Machine learning clásico",
        "description": (
            "Regresión lineal y logística, árboles de decisión, SVM y k-NN. "
            "Fundamentos prácticos de aprendizaje supervisado con scikit-learn."
        ),
        "duration_hours": 6,
        "prerequisites": ["numpy_pandas", "linear_algebra", "probability"]
    },
    {
        "id": "optimization",
        "name": "Optimización matemática",
        "description": (
            "Descenso del gradiente, momentum, Adam y variantes. Cómo minimizar "
            "funciones de pérdida en la práctica, con y sin restricciones."
        ),
        "duration_hours": 4,
        "prerequisites": ["calculus", "linear_algebra"]
    },
    {
        "id": "text_preprocessing",
        "name": "Preprocesamiento de texto",
        "description": (
            "Tokenización, stemming, lematización, TF-IDF y representaciones bag-of-words. "
            "Paso previo indispensable para cualquier tarea de procesamiento de lenguaje."
        ),
        "duration_hours": 2,
        "prerequisites": ["python_basics"]
    },

    # ── Level 2: require level-1 ───────────────────────────────────────────
    {
        "id": "neural_networks",
        "name": "Redes neuronales densas",
        "description": (
            "Perceptrón multicapa, funciones de activación, backpropagation y "
            "regularización. La base del deep learning moderno."
        ),
        "duration_hours": 6,
        "prerequisites": ["classical_ml", "optimization"]
    },
    {
        "id": "model_evaluation",
        "name": "Evaluación y validación de modelos",
        "description": (
            "Métricas de clasificación y regresión, validación cruzada, curvas ROC "
            "y detección de overfitting. Cómo saber si un modelo realmente aprende."
        ),
        "duration_hours": 3,
        "prerequisites": ["classical_ml", "probability"]
    },
    {
        "id": "embeddings",
        "name": "Word embeddings",
        "description": (
            "Word2Vec, GloVe y FastText. Representaciones vectoriales densas de palabras "
            "que capturan relaciones semánticas. Fundamento de los modelos modernos de NLP."
        ),
        "duration_hours": 3,
        "prerequisites": ["text_preprocessing", "linear_algebra", "neural_networks"]
    },
    {
        "id": "cnn",
        "name": "Redes convolucionales (CNN)",
        "description": (
            "Capas convolucionales, pooling y arquitecturas para visión por computadora. "
            "Incluye ResNet y transfer learning para clasificación de imágenes."
        ),
        "duration_hours": 5,
        "prerequisites": ["neural_networks"]
    },
    {
        "id": "rnn",
        "name": "Redes recurrentes (RNN/LSTM)",
        "description": (
            "Modelado de secuencias con RNNs, el problema del gradiente evanescente "
            "y cómo lo resuelven las LSTM. Aplicaciones en series de tiempo y texto."
        ),
        "duration_hours": 5,
        "prerequisites": ["neural_networks", "embeddings"]
    },

    # ── Level 3: advanced models ───────────────────────────────────────────
    {
        "id": "transformers",
        "name": "Arquitectura Transformer",
        "description": (
            "Mecanismo de atención, atención multi-cabeza, encoders y decoders. "
            "La arquitectura detrás de BERT, GPT y todos los LLMs modernos."
        ),
        "duration_hours": 7,
        "prerequisites": ["rnn", "embeddings", "optimization"]
    },
    {
        "id": "llm_fine_tuning",
        "name": "Fine-tuning de LLMs",
        "description": (
            "Técnicas para adaptar modelos de lenguaje preentrenados a tareas específicas: "
            "full fine-tuning, LoRA y prompting avanzado. Uso de HuggingFace Transformers."
        ),
        "duration_hours": 6,
        "prerequisites": ["transformers", "model_evaluation"]
    },
    {
        "id": "classical_nlp",
        "name": "NLP clásico",
        "description": (
            "Análisis sintáctico, reconocimiento de entidades, clasificación de texto "
            "y análisis de sentimiento con spaCy y NLTK. Fundamentos antes de los LLMs."
        ),
        "duration_hours": 4,
        "prerequisites": ["text_preprocessing", "classical_ml"]
    },
    {
        "id": "basic_rl",
        "name": "Aprendizaje por refuerzo",
        "description": (
            "Agentes, entornos, recompensas y políticas. Q-learning y Policy Gradient. "
            "Base para RLHF, que es cómo se alinean los LLMs modernos."
        ),
        "duration_hours": 6,
        "prerequisites": ["optimization", "probability", "neural_networks"]
    },
    {
        "id": "mlops",
        "name": "MLOps y despliegue",
        "description": (
            "Ciclo de vida de modelos en producción: pipelines, versionado con MLflow, "
            "contenedores Docker y APIs con FastAPI. Llevar un modelo del notebook al mundo real."
        ),
        "duration_hours": 5,
        "prerequisites": ["model_evaluation", "numpy_pandas", "python_basics"]
    },
    {
        "id": "interpretability",
        "name": "Interpretabilidad de modelos",
        "description": (
            "SHAP, LIME y técnicas de explicabilidad para entender por qué un modelo "
            "toma una decisión. Clave en aplicaciones de alto riesgo como medicina o justicia."
        ),
        "duration_hours": 4,
        "prerequisites": ["classical_ml", "model_evaluation"]
    },
]


def validate_dataset(resources: list) -> bool:
    """
    Check internal consistency of the dataset:
    - No duplicate IDs
    - All prerequisites reference existing IDs
    - No cycles (DAG requirement: topological sort must be possible)
    """
    ids = {r["id"] for r in resources}

    if len(ids) != len(resources):
        print("ERROR: duplicate IDs found")
        return False

    for r in resources:
        for prereq in r["prerequisites"]:
            if prereq not in ids:
                print(f"ERROR: '{r['id']}' references unknown prerequisite '{prereq}'")
                return False

    # Cycle detection via DFS with an active stack
    visited = set()
    active_stack = set()

    def has_cycle(node_id: str) -> bool:
        visited.add(node_id)
        active_stack.add(node_id)
        resource = next(r for r in resources if r["id"] == node_id)
        for prereq in resource["prerequisites"]:
            if prereq not in visited:
                if has_cycle(prereq):
                    return True
            elif prereq in active_stack:
                print(f"ERROR: cycle detected at '{node_id}' -> '{prereq}'")
                return True
        active_stack.remove(node_id)
        return False

    for r in resources:
        if r["id"] not in visited:
            if has_cycle(r["id"]):
                return False

    print(f"Dataset valid: {len(resources)} resources, no cycles, no broken references.")
    return True


def print_stats(resources: list):
    """Print a summary of the dataset."""
    total_hours = sum(r["duration_hours"] for r in resources)
    roots = [r for r in resources if not r["prerequisites"]]
    multi_prereq = [r for r in resources if len(r["prerequisites"]) > 1]

    print(f"\nDataset statistics:")
    print(f"  Total resources       : {len(resources)}")
    print(f"  Total hours           : {total_hours}h")
    print(f"  Root resources (no prerequisites): {len(roots)} — {[r['id'] for r in roots]}")
    print(f"  Resources with 2+ prerequisites  : {len(multi_prereq)}")
    print(f"  Min duration          : {min(r['duration_hours'] for r in resources)}h")
    print(f"  Max duration          : {max(r['duration_hours'] for r in resources)}h")


if __name__ == "__main__":
    if validate_dataset(resources):
        print_stats(resources)
        with open("resources.json", "w", encoding="utf-8") as f:
            json.dump(resources, f, ensure_ascii=False, indent=2)
        print("\nFile 'resources.json' generated successfully.")
