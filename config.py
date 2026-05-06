# Modèle d'embedding — doit être identique à l'indexation et à la recherche
EMBEDDING_MODEL = "paraphrase-multilingual-mpnet-base-v2"

# Chemins vers la base vectorielle
INDEX_PATH  = "index/faiss.index"
CHUNKS_PATH = "index/chunks.json"

# Modèle LLM Groq
LLM_MODEL = "llama-3.3-70b-versatile"

# Seuil de pertinence pour FAISS (distance L2 — plus petit = plus pertinent)
SEUIL_CONFIANCE = 5.0

# Liste des médicaments du corpus
MEDICAMENTS = [
    "doliprane", "dafalgan", "efferalgan",
    "ibuprofene", "advil", "nurofen",
    "aspirine", "aspegic",
    "amoxicilline", "augmentin",
    "smecta", "imodium",
    "ventoline", "becotide",
    "omeprazole", "inexium",
    "metformine", "glucophage"
]